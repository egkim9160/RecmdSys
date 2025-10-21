#!/usr/bin/env python3
"""
OpenSearch → 후보 의사 전체 내보내기(CSV)

요구사항:
- OpenSearch 인덱스의 모든 의사 문서를 스캔하여 CSV로 저장
- vector_field(임베딩) 포함 저장(JSON 문자열)
- 필수 컬럼이 결측인 유저는 제외하고, 별도 CSV에 {U_ID, missing_columns} 저장
- 저장 시 지오코딩(네이버)으로 HOME_ADDR, OFFICE_ADDR 좌표를 계산해 함께 저장

출력 파일(기본 out_dir=data/candidates_export):
- candidates_raw.csv        : 스캔 원본(정규 컬럼 구성)
- candidates_ok.csv         : 필수 컬럼 충족 행만
- candidates_missing.csv    : 누락 컬럼 있는 유저 목록(U_ID, missing_columns)
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """Ensure '/SPO/Project/HLink/RecmdSys' (project root for this tool) is on sys.path.

    This allows imports like 'from module.xxx import ...' to work when running
    this file directly (e.g., `python tools/export_candidates_from_opensearch.py`).
    """
    try:
        here = Path(__file__).resolve()
        project_root = here.parent.parent  # RecmdSys/
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    except Exception:
        # Best-effort; don't block execution if this fails
        pass


# Ensure module package can be resolved before any dynamic imports inside functions
_ensure_project_root_on_path()


def _load_env() -> None:
    try:
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(str(env_path))
    except Exception:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass


def init_opensearch_client():
    from opensearchpy import OpenSearch
    import ssl
    import urllib3

    host = os.getenv("OPENSEARCH_HOST")
    port = int(os.getenv("OPENSEARCH_PORT", "443"))
    username = os.getenv("OPENSEARCH_ID") or os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PW") or os.getenv("OPENSEARCH_PASSWORD")

    if not host:
        raise RuntimeError("OPENSEARCH_HOST가 설정되지 않았습니다.")

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    pool_manager = urllib3.PoolManager(
        num_pools=10,
        maxsize=25,
        block=False,
        timeout=urllib3.util.Timeout(connect=3.0, read=30.0),
        retries=urllib3.util.Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]),
    )

    kwargs = dict(
        hosts=[{"host": host, "port": port}],
        use_ssl=True,
        verify_certs=False,
        ssl_context=ssl_context,
        ssl_show_warn=False,
        ssl_assert_hostname=False,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
        pool_maxsize=25,
        pool_block=False,
        pool_manager=pool_manager,
    )
    if isinstance(username, str) and username and isinstance(password, str) and password:
        kwargs["http_auth"] = (username, password)
    return OpenSearch(**kwargs)


def scan_all(client, index_name: str, page_size: int = 1000, max_docs: int = 0) -> List[Dict[str, Any]]:
    body = {
        "size": page_size,
        "_source": True,
        "query": {"match_all": {}},
        "track_total_hits": True,
    }
    res = client.search(index=index_name, body=body, scroll="2m")
    scroll_id = res.get("_scroll_id")
    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict[str, Any]] = []
    out.extend(hits)
    while True:
        if max_docs and len(out) >= max_docs:
            break
        if not hits:
            break
        res = client.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = res.get("_scroll_id")
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        out.extend(hits)
    try:
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass
    if max_docs:
        return out[:max_docs]
    return out


def geocode(addr: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """NAVER 지오코딩 → 실패 시 LLM으로 주소 정제 후 재시도 → 그래도 실패 시 None

    1) 직접 geocode
    2) 실패하면 LLM으로 주소 정제(module.llm_utils.clean_address_with_llm)
    3) 정제 주소로 geocode 재시도

    Note: Logging is suppressed here to avoid excessive logs for bulk geocoding.
          Failed addresses will be tracked separately.
    """
    import random
    debug_sample = False  # Suppress success/debug logs entirely

    if not isinstance(addr, str) or not addr.strip():
        if debug_sample:
            print(f"[GEOCODE_DEBUG] Empty/invalid address: {repr(addr)}")
        return None, None
    raw = addr.strip()

    if debug_sample:
        print(f"[GEOCODE_DEBUG] Starting geocode for: '{raw[:50]}...'")

    # First attempt: direct geocoding (suppress module-level logging)
    try:
        from module.naver_geo import geocode_naver
        import logging
        # Temporarily suppress naver_geo logging for bulk operations
        naver_logger = logging.getLogger('module.naver_geo')
        original_level = naver_logger.level
        naver_logger.setLevel(logging.ERROR)
        try:
            if debug_sample:
                print(f"[GEOCODE_DEBUG] Calling geocode_naver for: '{raw}'")
            lat, lon = geocode_naver(raw)
            if debug_sample:
                print(f"[GEOCODE_DEBUG] geocode_naver returned: lat={lat}, lon={lon}")
            if lat is not None and lon is not None:
                # Suppress success logs; return silently on success
                return lat, lon
            elif debug_sample:
                print(f"[GEOCODE_DEBUG] geocode_naver returned None values for: '{raw}'")
        finally:
            naver_logger.setLevel(original_level)
    except Exception as e:
        # Log first few errors for debugging
        if random.random() < 0.01:  # Log 1% of errors
            print(f"[GEOCODE_ERROR] Direct geocoding exception for '{raw}': {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        pass

    # Second attempt: LLM cleaning + geocoding (suppress module-level logging)
    try:
        from module.llm_utils import get_openai_client, clean_address_with_llm
        import logging
        llm_logger = logging.getLogger('module.llm_utils')
        naver_logger = logging.getLogger('module.naver_geo')
        original_llm = llm_logger.level
        original_naver = naver_logger.level
        llm_logger.setLevel(logging.ERROR)
        naver_logger.setLevel(logging.ERROR)
        try:
            if debug_sample:
                print(f"[GEOCODE_DEBUG] Attempting LLM cleaning for: '{raw}'")
            client = get_openai_client()
            if debug_sample:
                print(f"[GEOCODE_DEBUG] LLM client obtained: {client is not None}")
            if client is not None:
                cleaned = clean_address_with_llm(raw, client)
                if debug_sample:
                    print(f"[GEOCODE_DEBUG] LLM cleaned result: '{cleaned}'")
                if isinstance(cleaned, str) and cleaned.strip():
                    try:
                        from module.naver_geo import geocode_naver as _geocode
                        if debug_sample:
                            print(f"[GEOCODE_DEBUG] Calling geocode_naver for cleaned: '{cleaned.strip()}'")
                        lat2, lon2 = _geocode(cleaned.strip())
                        if debug_sample:
                            print(f"[GEOCODE_DEBUG] geocode_naver returned for cleaned: lat={lat2}, lon={lon2}")
                        if lat2 is not None and lon2 is not None:
                            # Suppress success logs; return silently on success
                            return lat2, lon2
                    except Exception as e:
                        if random.random() < 0.01:  # Log 1% of errors
                            print(f"[GEOCODE_ERROR] LLM geocoding exception for '{cleaned.strip()}': {type(e).__name__}: {e}")
                            import traceback
                            traceback.print_exc()
                        pass
        finally:
            llm_logger.setLevel(original_llm)
            naver_logger.setLevel(original_naver)
    except Exception as e:
        if debug_sample:
            print(f"[GEOCODE_DEBUG] LLM attempt exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        pass

    if debug_sample:
        print(f"[GEOCODE_DEBUG] All attempts failed for: '{raw}', returning (None, None)")
    return None, None


def normalize_career_years(val: Any) -> int:
    try:
        if val is None:
            return 0
        s = str(val).strip()
        if s == "":
            return 0
        # 숫자 우선 추출
        import re
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
        return 0
    except Exception:
        return 0


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Export all candidates from OpenSearch to CSV (with geocoding)")
    parser.add_argument("--index_name", type=str, default=os.getenv("INDEX_NAME_HL", ""))
    parser.add_argument("--vector_field", type=str, default=os.getenv("HLINK_VECTOR_FIELD", "vector_field"))
    parser.add_argument("--out_dir", type=str, default="data/candidates_export")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--page_size", type=int, default=1000)
    parser.add_argument("--max_docs", type=int, default=0, help="0=all")
    args = parser.parse_args()

    if not args.index_name:
        raise RuntimeError("--index_name 또는 INDEX_NAME_HL 환경변수가 필요합니다.")

    os.makedirs(args.out_dir, exist_ok=True)

    client = init_opensearch_client()
    hits = scan_all(client, args.index_name, page_size=int(args.page_size), max_docs=int(args.max_docs))
    if not hits:
        raise RuntimeError("OpenSearch 인덱스에서 문서를 찾지 못했습니다.")

    rows: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {})
        meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else src
        if not isinstance(meta, dict):
            meta = {}

        # vector 추출
        vec_raw = None
        vf = src.get(args.vector_field) if args.vector_field in src else src.get("vector_field")
        if isinstance(vf, dict) and isinstance(vf.get("vector"), list):
            vec_raw = vf.get("vector")
        elif isinstance(vf, list):
            vec_raw = vf

        row = {
            "U_ID": meta.get("U_ID"),
            "SPECIALTY": meta.get("SPECIALTY"),
            "HOME_ADDR": meta.get("HOME_ADDR"),
            "OFFICE_ADDR": meta.get("OFFICE_ADDR"),
            "CAREER_YEARS": meta.get("CAREER_YEARS"),
            "vector_field": json.dumps(vec_raw, ensure_ascii=False) if isinstance(vec_raw, list) else None,
        }
        rows.append(row)

    df = pd.DataFrame.from_records(rows)

    # CAREER_YEARS 정수화(최소 규칙)
    df["CAREER_YEARS"] = df["CAREER_YEARS"].apply(normalize_career_years)

    # 지오코딩: HOME_ADDR → geo_ADDR_lat/lon, OFFICE_ADDR → geo_OFFICE_ADDR_lat/lon
    workers = int(args.workers) if hasattr(args, 'workers') else 32  # Capture workers in local scope

    def _batch_geocode(addrs: List[str], addr_type: str) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], List[str]]:
        """
        Batch geocode addresses.
        Returns: (results_dict, failed_addresses_list)
        """
        results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        failed: List[str] = []
        if not addrs:
            return results, failed
        cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        def task(a: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
            try:
                if a in cache:
                    return a, cache[a]
                lat, lon = geocode(a)
                cache[a] = (lat, lon)
                return a, (lat, lon)
            except Exception as e:
                print(f"[GEOCODE_TASK_ERROR] Exception in task for '{a}': {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return a, (None, None)
        uniq = sorted(set([a.strip() for a in addrs if isinstance(a, str) and a.strip()]))
        print(f"[GEOCODE] Starting batch geocoding for {len(uniq)} unique {addr_type} addresses...")
        print(f"[GEOCODE] Using {workers} workers")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(task, a) for a in uniq]
            for i, f in enumerate(as_completed(futs), 1):
                a, coord = f.result()
                results[a] = coord
                if coord == (None, None):
                    failed.append(a)
                if i % 100 == 0:
                    print(f"[GEOCODE] Progress: {i}/{len(uniq)} addresses processed...")
        print(f"[GEOCODE] Completed {addr_type}: {len(uniq) - len(failed)}/{len(uniq)} succeeded, {len(failed)} failed")
        return results, failed

    home_map, home_failed = _batch_geocode(df["HOME_ADDR"].dropna().astype(str).tolist(), "HOME")
    office_map, office_failed = _batch_geocode(df["OFFICE_ADDR"].dropna().astype(str).tolist(), "OFFICE")

    df["geo_ADDR_lat"] = df["HOME_ADDR"].apply(lambda a: home_map.get(a.strip(), (None, None))[0] if isinstance(a, str) and a.strip() else None)
    df["geo_ADDR_lon"] = df["HOME_ADDR"].apply(lambda a: home_map.get(a.strip(), (None, None))[1] if isinstance(a, str) and a.strip() else None)
    df["geo_OFFICE_ADDR_lat"] = df["OFFICE_ADDR"].apply(lambda a: office_map.get(a.strip(), (None, None))[0] if isinstance(a, str) and a.strip() else None)
    df["geo_OFFICE_ADDR_lon"] = df["OFFICE_ADDR"].apply(lambda a: office_map.get(a.strip(), (None, None))[1] if isinstance(a, str) and a.strip() else None)

    # 저장 1: raw
    raw_path = os.path.join(args.out_dir, "candidates_raw.csv")
    df.to_csv(raw_path, index=False)

    # 필수 컬럼 검사
    required_cols = [
        "U_ID",
        "SPECIALTY",
        "CAREER_YEARS",
        "geo_ADDR_lat",
        "geo_ADDR_lon",
        "geo_OFFICE_ADDR_lat",
        "geo_OFFICE_ADDR_lon",
        "vector_field",
    ]
    missing_rows: List[Dict[str, Any]] = []
    ok_mask = []
    for _, r in df.iterrows():
        missing = [c for c in required_cols if r.get(c) in (None, "", float("nan"))]
        if missing:
            missing_rows.append({
                "U_ID": r.get("U_ID"),
                "missing_columns": ",".join(missing)
            })
            ok_mask.append(False)
        else:
            ok_mask.append(True)

    df_ok = df[ok_mask].copy()
    df_missing = pd.DataFrame(missing_rows)

    ok_path = os.path.join(args.out_dir, "candidates_ok.csv")
    miss_path = os.path.join(args.out_dir, "candidates_missing.csv")
    df_ok.to_csv(ok_path, index=False)
    df_missing.to_csv(miss_path, index=False)

    # Save failed geocoding addresses to separate file
    geocode_failed_path = os.path.join(args.out_dir, "geocode_failed.csv")
    failed_records = []
    for addr in set(home_failed):
        failed_records.append({"address": addr, "type": "HOME_ADDR"})
    for addr in set(office_failed):
        failed_records.append({"address": addr, "type": "OFFICE_ADDR"})
    if failed_records:
        pd.DataFrame(failed_records).to_csv(geocode_failed_path, index=False)
        print(f"[GEOCODE] Saved {len(failed_records)} failed geocoding addresses to: {geocode_failed_path}")

    print(json.dumps({
        "raw": os.path.abspath(raw_path),
        "ok": os.path.abspath(ok_path),
        "missing": os.path.abspath(miss_path),
        "geocode_failed": os.path.abspath(geocode_failed_path) if failed_records else None,
        "total_docs": int(len(df)),
        "ok_docs": int(len(df_ok)),
        "missing_docs": int(len(df_missing)),
        "geocode_failed_count": len(failed_records),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()


