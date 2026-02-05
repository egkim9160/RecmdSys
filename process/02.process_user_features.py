#!/usr/bin/env python3
import os
import sys
import re
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# Ensure project root (RecmdSys/) is on sys.path for absolute imports like `module.*`
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass

try:
    from module.naver_geo import geocode_naver  # type: ignore
except Exception as import_err:
    raise RuntimeError(f"Failed to import geocoding module from project module path: {import_err}")

from module.db_utils import get_connection


def load_environment() -> None:
    # Always load from project root (run_pipeline.py location)
    try:
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        # fallback: current working dir
        if os.path.exists(".env"):
            load_dotenv(".env")


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


def _batch_embed_texts_gemini(texts, verbose: bool = False, log_interval: int = 100, max_workers: int = 10):
    """Gemini embedding-001을 사용한 배치 임베딩 (3072d, SEMANTIC_SIMILARITY, 병렬 처리)"""
    from module.llm_utils import batch_embed_texts_gemini
    return batch_embed_texts_gemini(texts, task_type="SEMANTIC_SIMILARITY", verbose=verbose, log_interval=log_interval, max_workers=max_workers)


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


def fetch_hospital_hira_coords(hospital_codes: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    HOSPITAL_HIRA 테이블에서 ykiho(=U_HOSPITAL_CODE) 기반으로 좌표 조회

    Args:
        hospital_codes: U_HOSPITAL_CODE 목록

    Returns:
        Dict[hospital_code -> (lat, lon)] - xPos=경도, yPos=위도
    """
    if not hospital_codes:
        return {}

    result = {}
    try:
        conn = get_connection()
        # batch query (IN clause) - 1000개씩 청크
        chunk_size = 1000
        for i in range(0, len(hospital_codes), chunk_size):
            chunk = hospital_codes[i:i+chunk_size]
            placeholders = ", ".join(["%s"] * len(chunk))
            query = f"""
                SELECT ykiho, xPos, yPos
                FROM medigate.HOSPITAL_HIRA
                WHERE ykiho IN ({placeholders})
                  AND xPos IS NOT NULL AND yPos IS NOT NULL
                  AND xPos != '' AND yPos != ''
            """
            df = pd.read_sql(query, conn, params=chunk)
            for _, row in df.iterrows():
                ykiho = str(row["ykiho"]) if row["ykiho"] else ""
                try:
                    # xPos=경도(lon), yPos=위도(lat)
                    lon = float(row["xPos"])
                    lat = float(row["yPos"])
                    if lat != 0 and lon != 0:
                        result[ykiho] = (lat, lon)
                except (ValueError, TypeError):
                    pass
        conn.close()
    except Exception as e:
        print(f"[WARN] HOSPITAL_HIRA 좌표 조회 실패: {e}")

    return result


def print_step(step: int, total: int, desc: str):
    """단계별 진행 상황 출력"""
    print(f"\n{'='*60}")
    print(f"[STEP {step}/{total}] {desc}")
    print(f"{'='*60}")


def main():
    load_environment()

    TOTAL_STEPS = 5  # 전체 단계 수

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
    parser.add_argument("--embed_workers", type=int, default=10, help="임베딩 병렬 처리 워커 수 (Gemini)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="출력 디렉터리 경로. 미지정 시 /SPO/Project/RecSys/data/processed/user_features_{timestamp}",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("사용자 피처 처리 파이프라인 시작")
    print("="*60)

    # IO paths
    raw_csv = args.input
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.strip() or f"/SPO/Project/RecSys/data/processed/user_features_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 입력 파일: {raw_csv}")
    print(f"[INFO] 출력 디렉토리: {out_dir}")

    # Read
    print_step(1, TOTAL_STEPS, "데이터 로드 및 전공 분류 처리")
    df = pd.read_csv(raw_csv)
    print(f"  - 원본 레코드 수: {len(df):,}건")
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
        print(f"  - limit 적용 후: {len(df):,}건")

    # Specialty rule
    df["SPECIALTY"] = df.apply(lambda r: compute_specialty(r.get("MAJOR_SPECIALTY"), r.get("DETAIL_SPECIALTY")), axis=1)
    print(f"  - SPECIALTY 컬럼 생성 완료")

    # =========================================================================
    # STEP 2: 좌표 수집 - HOSPITAL_HIRA 우선, geocoding fallback
    # =========================================================================
    print_step(2, TOTAL_STEPS, "좌표 데이터 수집 (병원코드 DB 조회 + 지오코딩)")

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

    # Collect addresses and hospital codes
    primary_addr_by_idx: Dict[int, str] = {}
    office_addr_by_idx: Dict[int, str] = {}
    hospital_code_by_idx: Dict[int, str] = {}  # U_HOSPITAL_CODE 수집

    for idx in range(len(df)):
        row = df.iloc[idx]
        addr, _src = pick_primary_address(row)
        if addr:
            primary_addr_by_idx[idx] = addr
        office_addr = _to_str_safe(row.get("U_OFFICE_ADDR") or row.get("U_OFFICE_ADDRESS"))
        if office_addr:
            office_addr_by_idx[idx] = office_addr
        # U_HOSPITAL_CODE 수집 (office 좌표 우선조회용)
        hosp_code = _to_str_safe(row.get("U_HOSPITAL_CODE"))
        if hosp_code:
            hospital_code_by_idx[idx] = hosp_code

    unique_primary = sorted(set(primary_addr_by_idx.values()))
    unique_office = sorted(set(office_addr_by_idx.values()))
    unique_hosp_codes = list(set(hospital_code_by_idx.values()))

    print(f"  - 주소 통계:")
    print(f"    • 거주지 주소 (unique): {len(unique_primary):,}건")
    print(f"    • 직장 주소 (unique): {len(unique_office):,}건")
    print(f"    • 병원코드 (unique): {len(unique_hosp_codes):,}건")

    # -------------------------------------------------------------------------
    # STEP 2-1: HOSPITAL_HIRA에서 병원코드로 좌표 우선 조회 (직장 좌표용)
    # -------------------------------------------------------------------------
    print(f"\n  [2-1] HOSPITAL_HIRA 테이블에서 병원코드 기반 좌표 조회 중...")
    hosp_coords = fetch_hospital_hira_coords(unique_hosp_codes)
    hosp_found_count = len(hosp_coords)
    print(f"       → 조회 성공: {hosp_found_count:,}건 / {len(unique_hosp_codes):,}건")

    # 병원코드로 좌표를 찾은 idx 기록
    office_coords_from_hira: Dict[int, Tuple[float, float]] = {}
    office_needs_geocoding_idx: set = set()

    for idx, hosp_code in hospital_code_by_idx.items():
        if hosp_code in hosp_coords:
            office_coords_from_hira[idx] = hosp_coords[hosp_code]
        elif idx in office_addr_by_idx:
            # 병원코드로 못 찾았고, 주소가 있으면 geocoding 필요
            office_needs_geocoding_idx.add(idx)

    # 병원코드 없지만 주소가 있는 경우도 geocoding 대상
    for idx in office_addr_by_idx:
        if idx not in hospital_code_by_idx and idx not in office_coords_from_hira:
            office_needs_geocoding_idx.add(idx)

    office_geocode_addresses = sorted(set(office_addr_by_idx[idx] for idx in office_needs_geocoding_idx))
    print(f"       → 직장 좌표 HIRA 성공: {len(office_coords_from_hira):,}건")
    print(f"       → 직장 지오코딩 필요: {len(office_geocode_addresses):,}건 (fallback)")

    # -------------------------------------------------------------------------
    # STEP 2-2: 지오코딩 (거주지 + 직장 fallback)
    # -------------------------------------------------------------------------
    print(f"\n  [2-2] Naver 지오코딩 실행 중... (workers={args.workers})")

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

    def geocode_many(addresses, desc: str = ""):
        results: Dict[str, Tuple[Optional[float], Optional[float], str, str]] = {}
        if not addresses:
            print(f"       → {desc} 대상 없음 (0건)")
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
                    success = sum(1 for v in results.values() if v[0] is not None)
                    print(f"       → {desc} 진행: {done:,}/{len(addresses):,}건 ({elapsed:.1f}s, 성공 {success:,}건)")
        elapsed = time.time() - start
        success = sum(1 for v in results.values() if v[0] is not None)
        print(f"       → {desc} 완료: {len(addresses):,}건 ({elapsed:.1f}s, 성공 {success:,}건)")
        return results

    print(f"\n       [거주지 주소 지오코딩]")
    primary_results = geocode_many(unique_primary, desc="거주지")

    print(f"\n       [직장 주소 지오코딩 - fallback]")
    office_results = geocode_many(office_geocode_addresses, desc="직장(fallback)")

    # -------------------------------------------------------------------------
    # STEP 2-3: 결과 병합 (HIRA 좌표 우선, geocoding fallback)
    # -------------------------------------------------------------------------
    print(f"\n  [2-3] 좌표 데이터 병합 중...")

    # 거주지 좌표 매핑
    primary_success = 0
    for idx, addr in primary_addr_by_idx.items():
        lat, lon, _status, src_used = primary_results.get(addr, (None, None, "fail", ""))
        df.at[idx, "geo_ADDR_src"] = src_used or addr
        df.at[idx, "geo_ADDR_lat"] = lat
        df.at[idx, "geo_ADDR_lon"] = lon
        if lat is not None:
            primary_success += 1

    # 직장 좌표 매핑: HOSPITAL_HIRA 우선, 없으면 geocoding 결과 사용
    office_hira_used = 0
    office_geo_used = 0
    office_failed = 0

    for idx in range(len(df)):
        # 1순위: HOSPITAL_HIRA 좌표
        if idx in office_coords_from_hira:
            lat, lon = office_coords_from_hira[idx]
            df.at[idx, "geo_OFFICE_ADDR_src"] = f"HIRA:{hospital_code_by_idx.get(idx, '')}"
            df.at[idx, "geo_OFFICE_ADDR_lat"] = lat
            df.at[idx, "geo_OFFICE_ADDR_lon"] = lon
            office_hira_used += 1
        # 2순위: geocoding 결과
        elif idx in office_addr_by_idx:
            addr = office_addr_by_idx[idx]
            lat, lon, _status, src_used = office_results.get(addr, (None, None, "fail", ""))
            df.at[idx, "geo_OFFICE_ADDR_src"] = src_used or addr
            df.at[idx, "geo_OFFICE_ADDR_lat"] = lat
            df.at[idx, "geo_OFFICE_ADDR_lon"] = lon
            if lat is not None:
                office_geo_used += 1
            else:
                office_failed += 1

    print(f"       → 거주지 좌표 성공: {primary_success:,}건 / {len(primary_addr_by_idx):,}건")
    print(f"       → 직장 좌표 성공 (HIRA): {office_hira_used:,}건")
    print(f"       → 직장 좌표 성공 (지오코딩): {office_geo_used:,}건")
    print(f"       → 직장 좌표 실패: {office_failed:,}건")

    # =========================================================================
    # STEP 3: 경력 연차 정규화
    # =========================================================================
    print_step(3, TOTAL_STEPS, "경력 연차 정규화")

    if "CAREER_YEARS" in df.columns:
        df["CAREER_YEARS"] = df["CAREER_YEARS"].apply(normalize_career_years)
    else:
        df["CAREER_YEARS"] = 0
    print(f"  - CAREER_YEARS 컬럼 정규화 완료")
    print(f"  - 경력 분포: 평균 {df['CAREER_YEARS'].mean():.1f}년, 최대 {df['CAREER_YEARS'].max()}년")

    # =========================================================================
    # STEP 4: 이력서 임베딩 (Gemini embedding-001, 3072d)
    # =========================================================================
    print_step(4, TOTAL_STEPS, "이력서 텍스트 임베딩 생성 (Gemini embedding-001, 3072d)")

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

    print(f"  - 임베딩 대상 텍스트: {len(texts):,}건")
    print(f"  - 모델: Gemini embedding-001 (3072 차원, SEMANTIC_SIMILARITY)")
    print(f"  - 병렬 처리: {args.embed_workers} workers")
    print(f"  - 긴 텍스트(>2000 토큰)는 자동 요약 후 임베딩")

    # 임베딩은 오래 걸리므로 항상 진행 로그 출력
    emb_vecs = _batch_embed_texts_gemini(texts, verbose=True, log_interval=100, max_workers=args.embed_workers)

    included_df["RESUME_EMB_3072"] = pd.NA
    emb_success = 0
    for idx, vec in enumerate(emb_vecs):
        if vec is not None:
            try:
                included_df.iat[idx, included_df.columns.get_loc("RESUME_EMB_3072")] = json.dumps(vec, ensure_ascii=False)
                emb_success += 1
            except Exception:
                included_df.iat[idx, included_df.columns.get_loc("RESUME_EMB_3072")] = pd.NA

    print(f"\n  - 임베딩 성공: {emb_success:,}건 / {len(texts):,}건")

    # Merge geocoding results from df into included_df (preserve original by index)
    geo_cols = [
        "geo_ADDR_lat", "geo_ADDR_lon",
        "geo_OFFICE_ADDR_lat", "geo_OFFICE_ADDR_lon",
        "geo_ADDR_src", "geo_OFFICE_ADDR_src",
    ]
    if args.verbose:
        for col in ["geo_ADDR_lat", "geo_ADDR_lon"]:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"[DEBUG] df.{col}: {non_null}/{len(df)} non-null values")

    for col in geo_cols:
        if col in df.columns:
            included_df.loc[:, col] = df.loc[included_df.index, col]

    if args.verbose:
        for col in ["geo_ADDR_lat", "geo_ADDR_lon"]:
            if col in included_df.columns:
                non_null = included_df[col].notna().sum()
                print(f"[DEBUG] included_df.{col}: {non_null}/{len(included_df)} non-null values")

    # =========================================================================
    # STEP 5: 결과 저장
    # =========================================================================
    print_step(5, TOTAL_STEPS, "처리 결과 저장")

    # Select required columns only (Gemini 3072d embedding 사용)
    keep_cols = [
        "U_ID",
        "SPECIALTY",
        "geo_ADDR_lat",
        "geo_ADDR_lon",
        "geo_OFFICE_ADDR_lat",
        "geo_OFFICE_ADDR_lon",
        "CAREER_YEARS",
        "BOARD_IDX",
        "RESUME_EMB_3072",
        "U_WORK_TYPE",
        "U_ORG_TYPE",
    ]
    # Ensure missing columns exist in included_df
    for c in keep_cols:
        if c not in included_df.columns:
            included_df[c] = pd.NA
    # 최종 결과
    out_df = included_df[keep_cols]

    # Save outputs
    out_csv = os.path.join(out_dir, "user_features_processed.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  - 처리된 레코드: {len(out_df):,}건")

    # Write notes
    notes_path = os.path.join(out_dir, "NOTES.txt")
    notes = []
    notes.append("가공 규칙:\n")
    notes.append("1) SPECIALTY 결정 규칙\n")
    notes.append("   - MAJOR_SPECIALTY = '내과' 이고 DETAIL_SPECIALTY 비어있지 않고 '세부분과없음'이 아닌 경우 -> SPECIALTY = DETAIL_SPECIALTY\n")
    notes.append("   - MAJOR_SPECIALTY = '내과' 이고 DETAIL_SPECIALTY 비어있음 또는 '세부분과없음' -> SPECIALTY = '내과'\n")
    notes.append("   - 그 외 -> SPECIALTY = MAJOR_SPECIALTY\n\n")
    notes.append("2) 지오코딩 (좌표 수집)\n")
    notes.append("   - geo_ADDR: R_ADDRESS 사용, 없으면 U_HOME_ADDR/U_HOME_ADDRESS 사용 → Naver 지오코딩\n")
    notes.append("   - geo_OFFICE_ADDR:\n")
    notes.append("     a) 1순위: U_HOSPITAL_CODE → HOSPITAL_HIRA.ykiho JOIN → xPos/yPos 좌표\n")
    notes.append("     b) 2순위(fallback): U_OFFICE_ADDR → Naver 지오코딩\n")
    notes.append("   - NAVER 지오코딩 실패 시 LLM으로 주소 정제(건물명 제거, 지번/도로명+번호만 유지, 행정구역 공식화) 후 1회 재시도\n")
    notes.append("   - 여전히 실패 시 빈 좌표로 둠\n\n")
    notes.append("3) 임베딩\n")
    notes.append("   - Gemini embedding-001 (3072d, SEMANTIC_SIMILARITY)\n")
    notes.append("   - 2000 토큰 초과 시 자동 요약 후 임베딩\n\n")
    notes.append("4) 산출 컬럼\n")
    notes.append("   - geo_ADDR_src, geo_ADDR_lat, geo_ADDR_lon\n")
    notes.append("   - geo_OFFICE_ADDR_src, geo_OFFICE_ADDR_lat, geo_OFFICE_ADDR_lon (HIRA:ykiho 또는 주소)\n")
    notes.append("   - RESUME_EMB_3072\n\n")
    notes.append(f"생성시각: {ts}\n")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("".join(notes))

    print(f"  - CSV 저장: {out_csv}")
    print(f"  - NOTES 저장: {notes_path}")

    # 최종 요약
    print("\n" + "="*60)
    print("사용자 피처 처리 완료!")
    print("="*60)
    print(f"  • 처리된 레코드: {len(out_df):,}건")
    print(f"  • 거주지 좌표 보유: {out_df['geo_ADDR_lat'].notna().sum():,}건")
    print(f"  • 직장 좌표 보유: {out_df['geo_OFFICE_ADDR_lat'].notna().sum():,}건")
    print(f"  • 임베딩 보유: {out_df['RESUME_EMB_3072'].notna().sum():,}건")
    print(f"  • 출력 경로: {out_dir}")


if __name__ == "__main__":
    main()


