#!/usr/bin/env python3
"""
HLINK end-to-end 파이프라인(한 번에 실행)

순서:
1) OpenSearch → 후보 의사 전체 내보내기(+vector, 지오코딩) → candidates_ok.csv
2) HLINK_IDX 공고 1건 로딩 + 텍스트 임베딩
3) 로컬 candidates_ok.csv 기반 유사도/거리 계산 → 04단계 스키마 CSV(step04.csv)
4) 모델 추론 → score, pred_1/0 추가된 CSV(scored.csv)

필요 환경:
- DB: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
- OpenSearch: OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_ID/OPENSEARCH_PW(옵션)
- Embedding: OPENAI_API_KEY (+ OPENAI_BASE_URL 또는 HLINK_EMBEDDING_BASE_URL)
"""

import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _ensure_project_root_on_path() -> None:
    try:
        here = Path(__file__).resolve()
        project_root = here.parent.parent  # RecmdSys/
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    except Exception:
        pass


def _load_env() -> None:
    try:
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


# -----------------------------
# Step 1: Export candidates
# -----------------------------
def export_candidates(index_name: str, out_dir: str, vector_field: str = "vector_field", workers: int = 32, page_size: int = 1000, max_docs: int = 0) -> Dict[str, Any]:
    from tools.export_candidates_from_opensearch import init_opensearch_client, scan_all, geocode, normalize_career_years

    client = init_opensearch_client()
    hits = scan_all(client, index_name=index_name, page_size=int(page_size), max_docs=int(max_docs))
    if not hits:
        raise RuntimeError("OpenSearch 인덱스에서 문서를 찾지 못했습니다.")

    rows: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {})
        meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else src
        if not isinstance(meta, dict):
            meta = {}
        vf = src.get(vector_field) if vector_field in src else src.get("vector_field")
        vec_raw = vf.get("vector") if isinstance(vf, dict) and isinstance(vf.get("vector"), list) else (vf if isinstance(vf, list) else None)
        rows.append({
            "U_ID": meta.get("U_ID"),
            "SPECIALTY": meta.get("SPECIALTY"),
            "HOME_ADDR": meta.get("HOME_ADDR"),
            "OFFICE_ADDR": meta.get("OFFICE_ADDR"),
            "CAREER_YEARS": meta.get("CAREER_YEARS"),
            "vector_field": json.dumps(vec_raw, ensure_ascii=False) if isinstance(vec_raw, list) else None,
        })

    df = pd.DataFrame.from_records(rows)
    # normalize career years (최소 규칙)
    df["CAREER_YEARS"] = df["CAREER_YEARS"].apply(normalize_career_years)

    # Geocoding
    def _batch_geocode(addrs: List[str], addr_type: str) -> Tuple[Dict[str, Tuple[Optional[float], Optional[float]]], List[str]]:
        """
        Batch geocode addresses with suppressed logging.
        Returns: (results_dict, failed_addresses_list)
        """
        results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        failed: List[str] = []
        if not addrs:
            return results, failed
        cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        uniq = sorted(set([a.strip() for a in addrs if isinstance(a, str) and a.strip()]))

        # Temporarily suppress module logging for bulk operations
        import logging
        naver_logger = logging.getLogger('module.naver_geo')
        llm_logger = logging.getLogger('module.llm_utils')
        original_naver = naver_logger.level
        original_llm = llm_logger.level
        naver_logger.setLevel(logging.ERROR)
        llm_logger.setLevel(logging.ERROR)

        logger.info(f"[GEOCODE] Starting batch geocoding for {len(uniq)} unique {addr_type} addresses...")
        def task(a: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
            if a in cache:
                return a, cache[a]

            # Keep external module loggers suppressed during geocode call
            lat, lon = geocode(a)

            cache[a] = (lat, lon)
            return a, (lat, lon)
        try:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                futs = [ex.submit(task, a) for a in uniq]
                for i, f in enumerate(as_completed(futs), 1):
                    a, coord = f.result()
                    results[a] = coord
                    if coord == (None, None):
                        failed.append(a)
                    if i % 100 == 0:
                        logger.info(f"[GEOCODE] Progress: {i}/{len(uniq)} addresses processed...")
        finally:
            naver_logger.setLevel(original_naver)
            llm_logger.setLevel(original_llm)

        logger.info(f"[GEOCODE] Completed {addr_type}: {len(uniq) - len(failed)}/{len(uniq)} succeeded, {len(failed)} failed")
        return results, failed

    home_map, home_failed = _batch_geocode(df["HOME_ADDR"].dropna().astype(str).tolist(), "HOME")
    office_map, office_failed = _batch_geocode(df["OFFICE_ADDR"].dropna().astype(str).tolist(), "OFFICE")

    df["geo_ADDR_lat"] = df["HOME_ADDR"].apply(lambda a: home_map.get(a.strip(), (None, None))[0] if isinstance(a, str) and a.strip() else None)
    df["geo_ADDR_lon"] = df["HOME_ADDR"].apply(lambda a: home_map.get(a.strip(), (None, None))[1] if isinstance(a, str) and a.strip() else None)
    df["geo_OFFICE_ADDR_lat"] = df["OFFICE_ADDR"].apply(lambda a: office_map.get(a.strip(), (None, None))[0] if isinstance(a, str) and a.strip() else None)
    df["geo_OFFICE_ADDR_lon"] = df["OFFICE_ADDR"].apply(lambda a: office_map.get(a.strip(), (None, None))[1] if isinstance(a, str) and a.strip() else None)

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "candidates_raw.csv")
    df.to_csv(raw_path, index=False)

    required_cols = ["U_ID","SPECIALTY","CAREER_YEARS","geo_ADDR_lat","geo_ADDR_lon","geo_OFFICE_ADDR_lat","geo_OFFICE_ADDR_lon","vector_field"]
    miss_rows: List[Dict[str, Any]] = []
    ok_mask = []
    for _, r in df.iterrows():
        missing = [c for c in required_cols if r.get(c) in (None, "", float("nan"))]
        if missing:
            miss_rows.append({"U_ID": r.get("U_ID"), "missing_columns": ",".join(missing)})
            ok_mask.append(False)
        else:
            ok_mask.append(True)
    df_ok = df[ok_mask].copy()
    df_missing = pd.DataFrame(miss_rows)

    ok_path = os.path.join(out_dir, "candidates_ok.csv")
    miss_path = os.path.join(out_dir, "candidates_missing.csv")
    df_ok.to_csv(ok_path, index=False)
    df_missing.to_csv(miss_path, index=False)

    # Save failed geocoding addresses to separate file
    geocode_failed_path = os.path.join(out_dir, "geocode_failed.csv")
    failed_records = []
    for addr in set(home_failed):
        failed_records.append({"address": addr, "type": "HOME_ADDR"})
    for addr in set(office_failed):
        failed_records.append({"address": addr, "type": "OFFICE_ADDR"})
    if failed_records:
        pd.DataFrame(failed_records).to_csv(geocode_failed_path, index=False)
        logger.info(f"[GEOCODE] Saved {len(failed_records)} failed geocoding addresses to: {geocode_failed_path}")

    return {
        "raw": raw_path,
        "ok": ok_path,
        "missing": miss_path,
        "geocode_failed": geocode_failed_path if failed_records else None,
        "ok_docs": int(len(df_ok)),
        "total_docs": int(len(df)),
        "geocode_failed_count": len(failed_records)
    }


# -----------------------------
# Step 2~3: Build step04 from local candidates
# -----------------------------
def _get_db_conn():
    from module.db_utils import get_connection
    return get_connection()


def _clean_html(text_html: str) -> str:
    try:
        from module.html_utils import clean_html_and_get_urls
        t, _ = clean_html_and_get_urls(text_html)
        return t
    except Exception:
        return text_html


def _remove_qa_section(text: str) -> str:
    """Remove Q&A section that starts with 'Q.' from the text"""
    import re
    # Find the first occurrence of Q. (with possible whitespace/newlines before)
    # and remove everything from that point onwards
    pattern = r'[\s\n]*Q\.\s.*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _geocode_org(addr: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(addr, str) or not addr.strip():
        logger.warning(f"[GEOCODE] _geocode_org: Invalid address (empty or None)")
        return None, None
    logger.info(f"[GEOCODE] _geocode_org: Geocoding address='{addr.strip()}'")
    try:
        from module.naver_geo import geocode_naver
        lat, lon = geocode_naver(addr.strip())
        # Suppress success logs to reduce noise; only errors are logged elsewhere
        return lat, lon
    except Exception as e:
        logger.error(f"[GEOCODE] _geocode_org: Failed for address='{addr.strip()}': {e}")
        return None, None


def _embed_job_text(job_text: str) -> List[float]:
    from module.llm_utils import get_embedding_client
    logger.info("[EMBED] _embed_job_text: Getting embedding client")
    client = get_embedding_client()
    if client is None:
        logger.error("[EMBED] _embed_job_text: Failed to initialize embedding client")
        raise RuntimeError("임베딩 클라이언트를 초기화할 수 없습니다. OPENAI_API_KEY/OPENAI_BASE_URL 확인")
    model = os.getenv("HLINK_EMBEDDING_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"))

    # Add instruction for medical job posting embedding
    # This helps the model focus on medical specialties, skills, and professional requirements
    instruction = (
        "Represent this medical job posting for semantic search. "
        "Focus on: medical specialty (전문과), required clinical skills, "
        "experience level, work environment, and specific medical expertise needed."
    )

    # Prepend instruction to the text (some embedding models support this)
    instructed_text = f"{instruction}\n\n{job_text}"

    logger.info(f"[EMBED] _embed_job_text: Creating embedding with model={model}, text_length={len(job_text)}, with instruction")
    try:
        resp = client.embeddings.create(input=[instructed_text], model=model)
        embedding = list(resp.data[0].embedding)
        logger.info(f"[EMBED] _embed_job_text: Successfully created embedding (dimension={len(embedding)})")
        return embedding
    except Exception as e:
        logger.error(f"[EMBED] _embed_job_text: Error creating embedding: {e}", exc_info=True)
        raise


def _to_unit(v: List[float]) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(v, dtype=float)
        n = np.linalg.norm(arr)
        if n <= 0:
            return None
        return arr / n
    except Exception:
        return None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _extract_career_req_llm(clean_text: str) -> Tuple[str, Optional[int]]:
    """Extract career requirement from cleaned CONTENT using LLM.
    Returns (career_text, career_num) where career_text is '경력 무관' or numeric string,
    and career_num is an int or None when irrelevant.
    """
    logger.info("[LLM] _extract_career_req_llm: Extracting career requirement from job content")
    try:
        from module.llm_utils import get_openai_client
        cli = get_openai_client()
        if cli is None:
            logger.warning("[LLM] _extract_career_req_llm: OpenAI client is None, returning default")
            return ("경력 무관", None)
        sys_prompt = (
            "너는 채용 공고의 경력 요구사항만 추출하는 도우미야. 규칙:\n"
            "- 출력은 오직 다음 둘 중 하나:\n"
            "  1) '경력 무관'\n"
            "  2) 정수 숫자 하나(예: 3, 5, 7).\n"
            "- 해석 규칙:\n"
            "  · 'N년 이상' → N\n"
            "  · 'N~M년' → M\n"
            "  · 'N년 이하' → N\n"
            "  · 정보가 없거나 모호하면 '경력 무관'\n"
        )
        usr = f"다음 공고 본문에서 경력 요구를 추출해. 본문:\n{clean_text}\n정답:"
        logger.info(f"[LLM] _extract_career_req_llm: Sending request (content length: {len(clean_text)} chars)")
        resp = cli.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini"),
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr}],
        )
        ans = resp.choices[0].message.content.strip()
        logger.info(f"[LLM] _extract_career_req_llm: Response received: '{ans}'")
        if "무관" in ans:
            logger.info("[LLM] _extract_career_req_llm: Parsed as '경력 무관'")
            return ("경력 무관", None)
        try:
            n = int(float(ans))
            logger.info(f"[LLM] _extract_career_req_llm: Parsed as numeric: {n}")
            return (str(n), n)
        except Exception as e:
            logger.warning(f"[LLM] _extract_career_req_llm: Failed to parse '{ans}' as number: {e}")
            return ("경력 무관", None)
    except Exception as e:
        logger.error(f"[LLM] _extract_career_req_llm: Error during extraction: {e}", exc_info=True)
        return ("경력 무관", None)


def build_step04(hlink_idx: int, candidates_csv: str, out_step04_csv: str) -> str:
    logger.info(f"[STEP04] Starting build_step04 for HLINK_IDX={hlink_idx}")
    logger.info(f"[STEP04] Input candidates CSV: {candidates_csv}")
    logger.info(f"[STEP04] Output step04 CSV: {out_step04_csv}")

    # Load job
    logger.info(f"[SQL] Loading job data for HLINK_IDX={hlink_idx}")
    conn = _get_db_conn()
    try:
        sql = (
            "SELECT J.HLINK_IDX, J.TITLE, J.CONTENT, J.ORG_NAME, J.PAY, J.PAY_TYPE, J.SPC_CODE, J.SPC_DETAIL_CODE, "
            "       J.CLIENT_IDX, C.ADDR "
            "FROM medigate.HLINK_JOB J "
            "LEFT JOIN medigate.HLINK_CLIENT C ON C.CLIENT_IDX = J.CLIENT_IDX "
            "WHERE J.HLINK_IDX = %s AND J.DISPLAY_FLAG='Y' AND (J.DEL_FLAG IS NULL OR J.DEL_FLAG <> 'Y') LIMIT 1"
        )
        logger.info(f"[SQL] Executing query: {sql}")
        logger.info(f"[SQL] Parameters: hlink_idx={int(hlink_idx)}")
        job_df = pd.read_sql(sql, conn, params=(int(hlink_idx),))
        logger.info(f"[SQL] Query returned {len(job_df)} row(s)")
    finally:
        try:
            conn.close()
            logger.info("[SQL] Database connection closed")
        except Exception as e:
            logger.error(f"[SQL] Error closing connection: {e}")
    if job_df.empty:
        logger.error(f"[SQL] No job found for HLINK_IDX={hlink_idx}")
        raise RuntimeError(f"HLINK_IDX={hlink_idx} 공고를 찾지 못했습니다.")
    j = job_df.iloc[0].to_dict()
    logger.info(f"[JOB] Job data loaded: TITLE='{j.get('TITLE')}', ORG_NAME='{j.get('ORG_NAME')}'")
    title = (j.get("TITLE") or "").strip()
    content_clean = _clean_html(str(j.get("CONTENT") or ""))
    logger.info(f"[JOB] Content cleaned (length: {len(content_clean)} chars)")

    # Remove Q&A section
    content_clean = _remove_qa_section(content_clean)
    logger.info(f"[JOB] Q&A section removed (final length: {len(content_clean)} chars)")

    job_text = "\n\n".join([t for t in [title, content_clean] if t])

    logger.info("[EMBED] Starting job text embedding")
    job_vec = _embed_job_text(job_text)
    logger.info(f"[EMBED] Job text embedded (vector length: {len(job_vec)})")
    u_job = _to_unit(job_vec)
    if u_job is None:
        logger.error("[EMBED] Job embedding normalization failed")
        raise RuntimeError("공고 임베딩 정규화 실패")

    # org geocode (ADDR 우선, 없으면 ORG_NAME)
    org_name = str(j.get("ORG_NAME") or "").strip()
    addr = str(j.get("ADDR") or "").strip()
    addr_for_geo = addr if addr else org_name
    logger.info(f"[GEOCODE] Geocoding organization: addr='{addr}', org_name='{org_name}'")
    org_lat, org_lon = _geocode_org(addr_for_geo)
    # Suppress success coordinates log; keep errors only

    # career requirement from CONTENT via LLM
    career_text, career_num = _extract_career_req_llm(content_clean)
    logger.info(f"[CAREER] Career requirement extracted: text='{career_text}', num={career_num}")

    # 간단 SQL로 코드명 매핑 (board_meta 저장 전에 먼저 수행)
    code_names: List[str] = []
    try:
        logger.info(f"[SQL] Loading specialty code mappings for HLINK_IDX={hlink_idx}")
        conn2 = _get_db_conn()
        sql_map = (
            "SELECT GROUP_CONCAT(cm.CODE_NAME ORDER BY cm.CODE_NAME SEPARATOR ', ') AS SPECIALTIES_MAPPED "
            "FROM medigate.HLINK_JOB j "
            "JOIN medigate.CODE_MASTER cm ON cm.KBN='SPC' "
            " AND (FIND_IN_SET(cm.CODE, j.SPC_CODE) > 0 OR FIND_IN_SET(cm.CODE, j.SPC_DETAIL_CODE) > 0) "
            "WHERE j.HLINK_IDX = %s"
        )
        logger.info(f"[SQL] Executing specialty mapping query: {sql_map}")
        df_map = pd.read_sql(sql_map, conn2, params=(hlink_idx,))
        logger.info(f"[SQL] Specialty mapping query returned {len(df_map)} row(s)")
        mapped_str = str(df_map.iloc[0]["SPECIALTIES_MAPPED"]).strip() if (not df_map.empty and df_map.iloc[0]["SPECIALTIES_MAPPED"] is not None) else ""
        code_names = [t.strip() for t in mapped_str.split(',') if t.strip()]
        logger.info(f"[SQL] Specialty codes mapped: {code_names}")
    except Exception as e:
        logger.error(f"[SQL] Error loading specialty mappings: {e}", exc_info=True)
        code_names = []
    finally:
        try:
            conn2.close()
            logger.info("[SQL] Specialty mapping connection closed")
        except Exception as e:
            logger.error(f"[SQL] Error closing specialty mapping connection: {e}")

    # Board 메타 저장 (step04_csv 위치)
    logger.info("[BOARD_META] Preparing to save board_meta.csv")
    try:
        meta = {
            "HLINK_IDX": int(hlink_idx),
            "TITLE": j.get("TITLE"),
            "ORG_NAME": org_name,
            "CLIENT_IDX": j.get("CLIENT_IDX"),
            "ADDR": addr or None,
            "ORG_lat": float(org_lat) if org_lat is not None else None,
            "ORG_lon": float(org_lon) if org_lon is not None else None,
            "PAY": float(j.get("PAY")) if j.get("PAY") is not None and str(j.get("PAY")).strip() != "" else 0.0,
            "PAY_TYPE": j.get("PAY_TYPE"),
            "SPC_CODE": j.get("SPC_CODE"),
            "SPC_DETAIL_CODE": j.get("SPC_DETAIL_CODE"),
            # embedding inputs
            "EMBED_TITLE": title,
            "EMBED_CONTENT_CLEAN": content_clean,
            # mapped specialties
            "SPECIALTIES_MAPPED": ",".join(code_names) if code_names else None,
            # LLM parsed career
            "CAREER_TEXT": career_text,
            "CAREER_NUM": int(career_num) if isinstance(career_num, int) else None,
        }
        logger.info(f"[BOARD_META] Meta data prepared: {meta}")
        out_dir_for_board = os.path.dirname(os.path.abspath(out_step04_csv))
        os.makedirs(out_dir_for_board, exist_ok=True)
        board_meta_path = os.path.join(out_dir_for_board, "board_meta.csv")
        logger.info(f"[BOARD_META] Saving to: {board_meta_path}")
        pd.DataFrame([meta]).to_csv(board_meta_path, index=False)
        logger.info(f"[BOARD_META] Successfully saved board_meta.csv to {board_meta_path}")
    except Exception as e:
        logger.error(f"[BOARD_META] FAILED to save board_meta.csv: {e}", exc_info=True)
        # Re-raise to make the error visible
        raise RuntimeError(f"board_meta.csv 저장 실패: {e}") from e

    # candidates
    logger.info(f"[CANDIDATES] Loading candidates from {candidates_csv}")
    dfc = pd.read_csv(candidates_csv)
    logger.info(f"[CANDIDATES] Loaded {len(dfc)} candidate records")
    if dfc.empty:
        logger.error("[CANDIDATES] candidates_csv is empty")
        raise RuntimeError("candidates_csv가 비어 있습니다.")

    # similarity + distances
    recs: List[Dict[str, Any]] = []
    def _norm_spec(raw: Optional[str]) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        return "내과" if "내과" in s else s

    job_specs = set([_norm_spec(x) for x in code_names if _norm_spec(x)])
    logger.info(f"[STEP04] Job specialties for matching: {job_specs}")

    # First pass: calculate all distances
    temp_distances_home = []
    temp_distances_office = []

    for _, r in dfc.iterrows():
        uid = str(r.get("U_ID"))
        try:
            vec = json.loads(r.get("vector_field")) if isinstance(r.get("vector_field"), str) else None
        except Exception:
            vec = None
        u_vec = _to_unit(vec) if isinstance(vec, list) else None
        if u_vec is None:
            continue

        # Calculate distances
        d_home = d_office = -1
        try:
            h_lat, h_lon = float(r.get("geo_ADDR_lat")), float(r.get("geo_ADDR_lon"))
            o_lat, o_lon = float(r.get("geo_OFFICE_ADDR_lat")), float(r.get("geo_OFFICE_ADDR_lon"))
            if org_lat is not None and org_lon is not None:
                def _hav(a1, b1, a2, b2):
                    R = 6371.0088
                    import math
                    p1, p2 = math.radians(a1), math.radians(a2)
                    dp, dl = math.radians(a2 - a1), math.radians(b2 - b1)
                    x = (math.sin(dp/2)**2) + math.cos(p1)*math.cos(p2)*(math.sin(dl/2)**2)
                    return int(round(2 * R * math.atan2(math.sqrt(x), math.sqrt(1-x))))
                if not pd.isna(h_lat) and not pd.isna(h_lon):
                    d_home = _hav(h_lat, h_lon, org_lat, org_lon)
                if not pd.isna(o_lat) and not pd.isna(o_lon):
                    d_office = _hav(o_lat, o_lon, org_lat, org_lon)
        except Exception:
            pass

        if d_home != -1:
            temp_distances_home.append(d_home)
        if d_office != -1:
            temp_distances_office.append(d_office)

    # Calculate medians for missing values
    median_home = int(np.median(temp_distances_home)) if temp_distances_home else 50
    median_office = int(np.median(temp_distances_office)) if temp_distances_office else 50
    logger.info(f"[DISTANCE] Calculated medians - home: {median_home}km, office: {median_office}km")
    logger.info(f"[DISTANCE] Valid distances - home: {len(temp_distances_home)}/{len(dfc)}, office: {len(temp_distances_office)}/{len(dfc)}")

    # Second pass: build records with median replacement
    for _, r in dfc.iterrows():
        uid = str(r.get("U_ID"))
        try:
            vec = json.loads(r.get("vector_field")) if isinstance(r.get("vector_field"), str) else None
        except Exception:
            vec = None
        u_vec = _to_unit(vec) if isinstance(vec, list) else None
        if u_vec is None:
            continue
        sim = _cos(u_job, u_vec)

        # distances (정수 km)
        d_home = d_office = -1
        try:
            h_lat, h_lon = float(r.get("geo_ADDR_lat")), float(r.get("geo_ADDR_lon"))
            o_lat, o_lon = float(r.get("geo_OFFICE_ADDR_lat")), float(r.get("geo_OFFICE_ADDR_lon"))
            if org_lat is not None and org_lon is not None:
                def _hav(a1, b1, a2, b2):
                    R = 6371.0088
                    import math
                    p1, p2 = math.radians(a1), math.radians(a2)
                    dp, dl = math.radians(a2 - a1), math.radians(b2 - b1)
                    x = (math.sin(dp/2)**2) + math.cos(p1)*math.cos(p2)*(math.sin(dl/2)**2)
                    return int(round(2 * R * math.atan2(math.sqrt(x), math.sqrt(1-x))))
                if not pd.isna(h_lat) and not pd.isna(h_lon):
                    d_home = _hav(h_lat, h_lon, org_lat, org_lon)
                if not pd.isna(o_lat) and not pd.isna(o_lon):
                    d_office = _hav(o_lat, o_lon, org_lat, org_lon)
        except Exception:
            pass

        # Replace -1 with median
        if d_home == -1:
            d_home = median_home
        if d_office == -1:
            d_office = median_office

        spec = _norm_spec(r.get("SPECIALTY"))
        spec_match = 1 if (spec and spec in job_specs) else 0
        # CAREER 매칭: 공고 LLM 파싱 결과(career_num)와 의사 경력 비교
        try:
            raw_cy = r.get("CAREER_YEARS")
            c_years = int(float(raw_cy)) if str(raw_cy).strip() != "" else 0
        except Exception:
            import re
            m = re.search(r"(\d+)", str(r.get("CAREER_YEARS")))
            c_years = int(m.group(1)) if m else 0
        career_match = 1
        career_gap = 0
        is_career_irrelevant = 1
        if isinstance(career_num, int):
            is_career_irrelevant = 0
            career_match = 1 if c_years >= int(career_num) else 0
            career_gap = 0 if career_match == 1 else max(0, int(career_num) - c_years)
        recs.append({
            "doctor_id": uid,
            "board_id": int(hlink_idx),
            "spec_match": int(spec_match),
            "distance_home": int(d_home),
            "distance_office": int(d_office),
            "CAREER_YEARS": int(c_years),
            "PAY": float(j.get("PAY")) if j.get("PAY") is not None and str(j.get("PAY")).strip() != "" else 0.0,
            "ORG_lat": float(org_lat) if org_lat is not None else None,
            "ORG_lon": float(org_lon) if org_lon is not None else None,
            "career_match": int(career_match),
            "career_gap": int(career_gap),
            "is_career_irrelevant": int(is_career_irrelevant),
            "similarity": float(sim),
            "applied": 0,
        })

    if not recs:
        logger.error("[STEP04] No records generated")
        raise RuntimeError("step04 레코드가 생성되지 않았습니다.")
    logger.info(f"[STEP04] Generated {len(recs)} candidate records")
    out_df = pd.DataFrame.from_records(recs)
    out_df = out_df.reset_index(drop=True)
    out_df.insert(0, "index", out_df.index.astype(int))
    cols = [
        "index","doctor_id","board_id","spec_match","distance_home","distance_office",
        "CAREER_YEARS","PAY","ORG_lat","ORG_lon","career_match","career_gap","is_career_irrelevant","similarity","applied"
    ]
    out_df = out_df[cols]
    os.makedirs(os.path.dirname(os.path.abspath(out_step04_csv)), exist_ok=True)
    logger.info(f"[STEP04] Saving step04 CSV to: {out_step04_csv}")
    out_df.to_csv(out_step04_csv, index=False)
    logger.info(f"[STEP04] Successfully saved step04 CSV with {len(out_df)} records")
    return out_step04_csv


# -----------------------------
# Step 4: Inference
# -----------------------------
def run_inference(model_json: str, features_json: Optional[str], input_csv: str, output_csv: str) -> str:
    from tools.infer_xgb import load_feature_columns, load_xgb_booster, predict_proba_with_booster
    feats = load_feature_columns(model_json, features_json or None)
    booster = load_xgb_booster(model_json)
    df = pd.read_csv(input_csv)
    y_prob = predict_proba_with_booster(booster, df, feats)
    y_pred_1 = (y_prob >= 0.5).astype(int)
    y_pred_0 = 1 - y_pred_1
    out = df.copy()
    out["score"] = y_prob
    out["pred_1"] = y_pred_1
    out["pred_0"] = y_pred_0
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    _ensure_project_root_on_path()
    _load_env()

    p = argparse.ArgumentParser(description="HLINK end-to-end pipeline (export → step04 → infer)")
    p.add_argument("--hlink_idx", type=int, required=True)
    p.add_argument("--index_name", type=str, default=os.getenv("INDEX_NAME_HL", ""))
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--vector_field", type=str, default=os.getenv("HLINK_VECTOR_FIELD", "vector_field"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--page_size", type=int, default=1000)
    p.add_argument("--max_docs", type=int, default=0)
    p.add_argument("--model_json", type=str, required=True)
    p.add_argument("--features_json", type=str, default="")
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--addr_km", type=int, default=30, help="주소 매칭 임계(km). 이하이면 매칭으로 간주")
    p.add_argument("--candidates_csv", type=str, default="", help="사전 준비된 candidates_ok.csv 경로")
    args = p.parse_args()

    # 외부 candidates_csv를 사용하지 않고(export 수행 예정) 인덱스명이 없으면 에러
    if not args.index_name and not args.skip_export and not args.candidates_csv:
        raise RuntimeError("--index_name 또는 INDEX_NAME_HL 환경변수가 필요합니다.")

    # 기본 저장 위치를 /SPO/Project/HLink/RecmdSys/test 하위로 고정
    out_dir = args.out_dir.strip() or f"/SPO/Project/HLink/RecmdSys/test/board_{args.hlink_idx}"
    os.makedirs(out_dir, exist_ok=True)

    # Configure file logging in addition to console logging
    log_path = os.path.join(out_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(file_handler)

    logger.info("="*80)
    logger.info(f"[PIPELINE] Starting HLINK inference pipeline for HLINK_IDX={args.hlink_idx}")
    logger.info(f"[PIPELINE] Output directory: {out_dir}")
    logger.info(f"[PIPELINE] Log file: {log_path}")
    logger.info(f"[PIPELINE] Arguments: {vars(args)}")
    logger.info("="*80)

    # Step 1: export
    candidates_dir = os.path.join(out_dir, "candidates")
    paths = {"ok": os.path.join(candidates_dir, "candidates_ok.csv")}

    # 외부 candidates CSV가 지정된 경우: 그대로 사용하고 export 단계 스킵
    external_candidates = args.candidates_csv.strip()
    if external_candidates:
        if not os.path.exists(external_candidates):
            logger.error(f"[CANDIDATES] File not found: {external_candidates}")
            raise FileNotFoundError(f"candidates_csv 파일을 찾을 수 없습니다: {external_candidates}")
        paths["ok"] = external_candidates
        logger.info(f"[CANDIDATES] Using external CSV: {paths['ok']}")
    else:
        if not args.skip_export or not os.path.exists(paths["ok"]):
            logger.info("[CANDIDATES] Starting export from OpenSearch")
            res = export_candidates(index_name=args.index_name, out_dir=candidates_dir, vector_field=args.vector_field, workers=args.workers, page_size=args.page_size, max_docs=args.max_docs)
            paths.update(res)
            logger.info(f"[CANDIDATES] Export completed: ok={paths.get('ok')}, total_docs={res.get('total_docs')}, ok_docs={res.get('ok_docs')}")
        else:
            logger.info(f"[CANDIDATES] Reusing existing candidates: {paths['ok']}")

    # Step 2~3: build step04
    step04_csv = os.path.join(out_dir, "step04.csv")
    logger.info("[STEP04] Starting step04 build (embedding + specialty map + career parse + distances)")
    step04_csv = build_step04(hlink_idx=int(args.hlink_idx), candidates_csv=paths["ok"], out_step04_csv=step04_csv)
    logger.info(f"[STEP04] step04 CSV saved: {step04_csv}")

    # Step 4: inference
    scored_csv = os.path.join(out_dir, "scored.csv")
    features_json = args.features_json.strip() or None
    logger.info("[INFERENCE] Starting model inference")
    scored_csv = run_inference(model_json=args.model_json, features_json=features_json, input_csv=step04_csv, output_csv=scored_csv)
    logger.info(f"[INFERENCE] Inference completed, scored CSV saved: {scored_csv}")

    logger.info("="*80)
    logger.info("[PIPELINE] Pipeline completed successfully!")
    logger.info("="*80)
    print(json.dumps({
        "candidates_ok": os.path.abspath(paths["ok"]),
        "step04_csv": os.path.abspath(step04_csv),
        "scored_csv": os.path.abspath(scored_csv),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()


