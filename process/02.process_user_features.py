#!/usr/bin/env python3
import os
import sys
import re
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from dotenv import load_dotenv


# Ensure we can import the NAVER geocoding utility from the sibling project
GEOCODE_MODULE_PATH = "/SPO/Project/RecSys/scripts/module/"
if GEOCODE_MODULE_PATH not in sys.path:
    sys.path.append(GEOCODE_MODULE_PATH)

try:
    from naver_geo import geocode_naver  # type: ignore
except Exception as import_err:
    raise RuntimeError(f"Failed to import geocoding module from {GEOCODE_MODULE_PATH}: {import_err}")


def load_environment() -> None:
    # Load environment variables (NAVER, OPENAI, etc.)
    # Use .env file in current directory first, fallback to parent
    if os.path.exists(".env"):
        load_dotenv(".env")
    elif os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def get_openai_client():
    """Create an OpenAI client if keys exist. Returns None if not configured."""
    from module.llm_utils import get_openai_client as get_client
    return get_client()


def get_embedding_client():
    from module.llm_utils import get_embedding_client as get_embed_client
    return get_embed_client()


def _batch_embed_texts(client, texts, model: str = "text-embedding-3-large", batch_size: int = 64):
    from module.llm_utils import batch_embed_texts
    return batch_embed_texts(client, texts, model, batch_size)


def clean_address_with_llm(raw_address: str, client) -> Optional[str]:
    """
    Use LLM to normalize a messy Korean address string:
      - Remove building/apartment names, floors, room numbers
      - Keep up to lot number or road-name + building number
      - Normalize ambiguous admin divisions to official names
      - Return ONLY the cleaned address as plain text
    Returns None if client is unavailable or on failure.
    """
    from module.llm_utils import clean_address_with_llm as clean_addr
    return clean_addr(raw_address, client)


def try_geocode(address: Optional[str], cache: Dict[str, Tuple[float, float]], *, sleep_sec: float = 0.0, lock: Optional[Lock] = None) -> Tuple[Optional[float], Optional[float], str]:
    """Attempt geocoding using NAVER. Use cache to reduce calls.
    Returns (lat, lon, status). status in {ok, empty, error}
    """
    from module.geo_utils import try_geocode as geo_try
    return geo_try(address, cache, sleep_sec=sleep_sec, lock=lock)


def _to_str_safe(value: Optional[str]) -> str:
    """Convert to string safely. Returns empty string for non-strings/NaN."""
    from module.data_utils import to_str_safe
    return to_str_safe(value)


def compute_specialty(major: Optional[str], detail: Optional[str]) -> Optional[str]:
    from module.data_utils import compute_specialty as compute_spec
    return compute_spec(major, detail)


def pick_primary_address(row: pd.Series) -> Tuple[Optional[str], str]:
    """Pick primary address for geo_ADDR: prefer R_ADDRESS, else U_HOME_ADDR/U_HOME_ADDRESS."""
    r_addr = _to_str_safe(row.get("R_ADDRESS"))
    home_addr = _to_str_safe(row.get("U_HOME_ADDR") or row.get("U_HOME_ADDRESS"))

    if r_addr:
        return r_addr, "R_ADDRESS"
    if home_addr:
        return home_addr, "U_HOME_ADDR"
    return None, ""


def normalize_career_years(value: Optional[str]) -> int:
    """Map textual career years to integer years.
    Examples: '1~2년'->1, '2~3년'->2, '5~6년'->5, '10년 이상'->10, '1년 미만'->0, '경력 없음'->0
    Default fallback is 0 when unrecognized.
    """
    from module.data_utils import normalize_career_years as normalize
    return normalize(value)


def main():
    load_environment()

    # CLI
    parser = argparse.ArgumentParser(description="Process user_features with specialty rules, geocoding, and resume embedding")
    parser.add_argument("--input", default="/SPO/Project/RecSys/data/raw/user_features.csv", help="Input CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between geocode calls")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based address cleaning")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for geocoding")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs")
    parser.add_argument("--resume_policy", type=str, default="fallback", choices=["exclude", "fallback", "zero"], help="RESUME_TEXT 공백 처리 정책")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-large", help="임베딩 모델명(OpenAI 호환)")
    parser.add_argument("--embed_batch", type=int, default=64, help="임베딩 배치 크기")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="출력 디렉터리 경로. 미지정 시 /SPO/Project/RecSys/data/processed/user_features_{timestamp}",
    )
    args = parser.parse_args()

    # IO paths
    raw_csv = args.input
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.strip() or f"/SPO/Project/RecSys/data/processed/user_features_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Read
    df = pd.read_csv(raw_csv)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    # Specialty rule
    df["SPECIALTY"] = df.apply(lambda r: compute_specialty(r.get("MAJOR_SPECIALTY"), r.get("DETAIL_SPECIALTY")), axis=1)

    # Geocoding - parallel unique address processing
    cache: Dict[str, Tuple[float, float]] = {}
    cache_lock = Lock()
    openai_client = None if args.no_llm else get_openai_client()

    # Prepare new columns (no status columns)
    df["geo_ADDR_src"] = ""
    df["geo_ADDR_lat"] = pd.NA
    df["geo_ADDR_lon"] = pd.NA

    df["geo_OFFICE_ADDR_src"] = ""
    df["geo_OFFICE_ADDR_lat"] = pd.NA
    df["geo_OFFICE_ADDR_lon"] = pd.NA

    # Collect addresses
    primary_addr_by_idx: Dict[int, str] = {}
    office_addr_by_idx: Dict[int, str] = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        addr, _src = pick_primary_address(row)
        if addr:
            primary_addr_by_idx[idx] = addr
        office_addr = _to_str_safe(row.get("U_OFFICE_ADDR") or row.get("U_OFFICE_ADDRESS"))
        if office_addr:
            office_addr_by_idx[idx] = office_addr

    unique_primary = sorted(set(primary_addr_by_idx.values()))
    unique_office = sorted(set(office_addr_by_idx.values()))

    print(f"[INFO] Unique addresses - primary: {len(unique_primary)}, office: {len(unique_office)}")

    def geocode_pipeline(address: str) -> Tuple[str, Optional[float], Optional[float], str, str]:
        # returns (original_addr, lat, lon, status, src_used)
        if not address:
            return address, None, None, "empty", ""

        # Cache read (thread-safe)
        with cache_lock:
            if address in cache:
                lat, lon = cache[address]
                return address, lat, lon, "direct", address

        lat, lon, status = try_geocode(address, cache, sleep_sec=args.sleep)
        if status == "ok":
            return address, lat, lon, "direct", address

        # LLM fallback
        if openai_client:
            cleaned = clean_address_with_llm(address, openai_client)
            if cleaned:
                lat2, lon2, status2 = try_geocode(cleaned, cache, sleep_sec=args.sleep)
                if status2 == "ok":
                    return address, lat2, lon2, "llm", cleaned
        return address, None, None, "fail", ""

    def geocode_many(addresses):
        results: Dict[str, Tuple[Optional[float], Optional[float], str, str]] = {}
        if not addresses:
            return results
        start = time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            fut_to_addr = {ex.submit(geocode_pipeline, a): a for a in addresses}
            done = 0
            for fut in as_completed(fut_to_addr):
                a = fut_to_addr[fut]
                try:
                    _a, lat, lon, status, src_used = fut.result()
                except Exception:
                    lat, lon, status, src_used = None, None, "fail", ""
                results[a] = (lat, lon, status, src_used)
                done += 1
                if done % 100 == 0:
                    elapsed = time.time() - start
                    print(f"[INFO] Processed {done}/{len(addresses)} {elapsed:.1f}s")
        elapsed = time.time() - start
        print(f"[INFO] Completed {len(addresses)} addresses in {elapsed:.1f}s (workers={args.workers})")
        return results

    primary_results = geocode_many(unique_primary)
    office_results = geocode_many(unique_office)

    # Map back to rows (no status columns; if 좌표 없으면 빈값 유지)
    for idx, addr in primary_addr_by_idx.items():
        lat, lon, _status, src_used = primary_results.get(addr, (None, None, "fail", ""))
        df.at[idx, "geo_ADDR_src"] = src_used or addr
        df.at[idx, "geo_ADDR_lat"] = lat
        df.at[idx, "geo_ADDR_lon"] = lon

    for idx, addr in office_addr_by_idx.items():
        lat, lon, _status, src_used = office_results.get(addr, (None, None, "fail", ""))
        df.at[idx, "geo_OFFICE_ADDR_src"] = src_used or addr
        df.at[idx, "geo_OFFICE_ADDR_lat"] = lat
        df.at[idx, "geo_OFFICE_ADDR_lon"] = lon

    # Derive CAREER_YEARS (numeric) from CAREER_YEARS (text)
    if "CAREER_YEARS" in df.columns:
        df["CAREER_YEARS"] = df["CAREER_YEARS"].apply(normalize_career_years)
    else:
        df["CAREER_YEARS"] = 0

    # Resume embedding (4096d): SPECIALTY + CAREER_YEARS + U_HOME_ADDR + RESUME_TEXT
    def _compose_embed_text(row) -> Optional[str]:
        parts = []
        spec = _to_str_safe(row.get("SPECIALTY"))
        if spec:
            parts.append(spec)
        cy = row.get("CAREER_YEARS")
        try:
            if cy is not None and str(cy).strip() != "":
                cy_val = int(cy) if isinstance(cy, (int, float)) and not pd.isna(cy) else None
                parts.append(f"경력 {cy_val}년" if cy_val is not None else str(cy))
        except Exception:
            pass
        home = _to_str_safe(row.get("U_HOME_ADDR") or row.get("U_HOME_ADDRESS"))
        if home:
            parts.append(home)
        rtxt = _to_str_safe(row.get("RESUME_TEXT"))
        if rtxt:
            parts.append(rtxt)
        text = " | ".join([p for p in parts if p])
        return text.strip() if text.strip() else None

    df["EMBED_TEXT"] = df.apply(_compose_embed_text, axis=1)
    included_df = df[df["EMBED_TEXT"].notna() & (df["EMBED_TEXT"].astype(str).str.strip() != "")].copy()

    texts = included_df["EMBED_TEXT"].astype(str).tolist()
    emb_client = get_embedding_client()
    emb_vecs = _batch_embed_texts(emb_client, texts, model=args.embed_model, batch_size=max(1, int(args.embed_batch)))

    included_df["RESUME_EMB_4096"] = pd.NA
    for idx, vec in enumerate(emb_vecs):
        if vec is not None:
            try:
                included_df.iat[idx, included_df.columns.get_loc("RESUME_EMB_4096")] = json.dumps(vec, ensure_ascii=False)
            except Exception:
                included_df.iat[idx, included_df.columns.get_loc("RESUME_EMB_4096")] = pd.NA

    # Select required columns only
    keep_cols = [
        "U_ID",
        "SPECIALTY",
        "geo_ADDR_lat",
        "geo_ADDR_lon",
        "geo_OFFICE_ADDR_lat",
        "geo_OFFICE_ADDR_lon",
        "CAREER_YEARS",
        "BOARD_IDX",
        "RESUME_EMB_4096",
    ]
    # Ensure missing columns exist
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA
    # 최종 결과: 네 요소 모두 없는 사용자는 제외
    out_df = included_df[keep_cols]

    # Save outputs
    out_csv = os.path.join(out_dir, "user_features_processed.csv")
    out_df.to_csv(out_csv, index=False)

    # Write notes
    notes_path = os.path.join(out_dir, "NOTES.txt")
    notes = []
    notes.append("가공 규칙:\n")
    notes.append("1) SPECIALTY 결정 규칙\n")
    notes.append("   - MAJOR_SPECIALTY = '내과' 이고 DETAIL_SPECIALTY 비어있지 않고 '세부분과없음'이 아닌 경우 -> SPECIALTY = DETAIL_SPECIALTY\n")
    notes.append("   - MAJOR_SPECIALTY = '내과' 이고 DETAIL_SPECIALTY 비어있음 또는 '세부분과없음' -> SPECIALTY = '내과'\n")
    notes.append("   - 그 외 -> SPECIALTY = MAJOR_SPECIALTY\n\n")
    notes.append("2) 지오코딩\n")
    notes.append("   - geo_ADDR: R_ADDRESS 사용, 없으면 U_HOME_ADDR/U_HOME_ADDRESS 사용\n")
    notes.append("   - geo_OFFICE_ADDR: U_OFFICE_ADDR/U_OFFICE_ADDRESS 사용\n")
    notes.append("   - NAVER 지오코딩 실패 시 LLM으로 주소 정제(건물명 제거, 지번/도로명+번호만 유지, 행정구역 공식화) 후 1회 재시도\n")
    notes.append("   - 여전히 실패 시 빈 좌표로 둠\n\n")
    notes.append("3) 산출 컬럼\n")
    notes.append("   - geo_ADDR_src, geo_ADDR_lat, geo_ADDR_lon, geo_ADDR_status\n")
    notes.append("   - geo_OFFICE_ADDR_src, geo_OFFICE_ADDR_lat, geo_OFFICE_ADDR_lon, geo_OFFICE_ADDR_status\n\n")
    notes.append(f"생성시각: {ts}\n")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("".join(notes))

    print(f"Saved: {out_csv}")
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()


