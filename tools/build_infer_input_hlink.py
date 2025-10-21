#!/usr/bin/env python3
"""
H-LINK용 단일 공고(board_idx) 기준 전 의사 후보에 대한 similarity 계산 입력 생성기

기능:
- MySQL: medigate.HLINK_JOB에서 board(=HLINK_IDX) 상세 텍스트(TITLE/DISPLAY_TITLE+CONTENT) 로드
- 텍스트 임베딩: OpenAI 호환 임베딩 엔드포인트(환경변수)로 job 텍스트 벡터화
- OpenSearch: 후보자 인덱스에서 전체 문서 스캔(_source에 vector_field 포함) 후 코사인 유사도 계산
- 결과 CSV 저장: doctor_id, board_id, similarity 및 일부 메타

주의:
- doctor 벡터는 OpenSearch에 저장된 vector_field를 사용(의사 데이터 임베딩 금지)
- 대량 문서 환경을 고려해 scroll API 사용
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def get_db_conn():
    from module.db_utils import get_connection
    return get_connection()


def fetch_job_text(conn, board_idx: int) -> Tuple[str, Dict[str, Any]]:
    """HLINK_JOB에서 공고 텍스트 구성용 필드 로드.
    반환: (job_text, raw_row_dict)
    """
    sql = """
SELECT
  HLINK_IDX,
  TITLE,
  DISPLAY_TITLE,
  CONTENT,
  ORG_NAME,
  START_DATE,
  END_DATE,
  PAY,
  PAY_TYPE,
  INCENTIVE_FLAG,
  SPC_CODE,
  SPC_DETAIL_CODE
FROM medigate.HLINK_JOB
WHERE HLINK_IDX = %s
  AND (DISPLAY_FLAG = 'Y')
  AND (RECRUIT_FLAG = 'Y')
  AND (DEL_FLAG IS NULL OR DEL_FLAG <> 'Y')
LIMIT 1
"""
    df = pd.read_sql(sql, conn, params=(int(board_idx),))
    if df.empty:
        raise RuntimeError(f"HLINK_JOB에서 BOARD_IDX={board_idx} 레코드를 찾지 못했습니다.")
    row = df.iloc[0].to_dict()

    # 텍스트 구성: DISPLAY_TITLE(또는 TITLE) + CONTENT(HTML 정제)
    title = str(row.get("DISPLAY_TITLE") or row.get("TITLE") or "").strip()
    content_html = (row.get("CONTENT") or "")
    try:
        from module.html_utils import clean_html_and_get_urls
        clean_text, _ = clean_html_and_get_urls(str(content_html))
    except Exception:
        clean_text = str(content_html)
    job_text = "\n\n".join([t for t in [title, clean_text] if t])
    return job_text, row


def embed_job_text(job_text: str) -> List[float]:
    from module.llm_utils import get_embedding_client
    client = get_embedding_client()
    if client is None:
        raise RuntimeError("임베딩 클라이언트를 초기화할 수 없습니다. OPENAI_API_KEY/OPENAI_BASE_URL 확인")
    model = os.getenv("HLINK_EMBEDDING_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"))
    resp = client.embeddings.create(input=[job_text], model=model)
    return list(resp.data[0].embedding)


def init_opensearch_client():
    """환경변수 기반 OpenSearch 클라이언트 초기화(LLM-HLINK util 독립)."""
    from opensearchpy import OpenSearch
    import ssl
    import urllib3

    host = os.getenv("OPENSEARCH_HOST")
    port = int(os.getenv("OPENSEARCH_PORT", "443"))
    username = os.getenv("OPENSEARCH_ID")
    password = os.getenv("OPENSEARCH_PW")

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

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(username, password),
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


def scan_all_candidates(client, index_name: str, vector_field: str = "vector_field", page_size: int = 1000) -> List[Dict[str, Any]]:
    """OpenSearch Scroll API로 전체 문서 스캔. vector_field 포함해서 가져옴."""
    # 초기 검색
    body = {
        "size": page_size,
        "_source": True,  # 모든 source 포함 (vector_field 포함)
        "query": {"match_all": {}},
        "track_total_hits": True,
    }
    results = client.search(index=index_name, body=body, scroll="1m")
    scroll_id = results.get("_scroll_id")
    hits = results.get("hits", {}).get("hits", [])
    all_hits: List[Dict[str, Any]] = []
    all_hits.extend(hits)

    while True:
        if not hits:
            break
        results = client.scroll(scroll_id=scroll_id, scroll="1m")
        scroll_id = results.get("_scroll_id")
        hits = results.get("hits", {}).get("hits", [])
        if not hits:
            break
        all_hits.extend(hits)

    # scroll clear
    try:
        if scroll_id:
            client.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return all_hits


def to_unit_vector(vec: List[float]) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(vec, dtype=float)
        n = np.linalg.norm(arr)
        if n <= 0:
            return None
        return arr / n
    except Exception:
        return None


def compute_cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def extract_meta(source: Dict[str, Any]) -> Dict[str, Any]:
    meta = {k: v for k, v in source.items() if k != "text"}
    # vector_field는 제거
    if "vector_field" in meta:
        meta.pop("vector_field", None)
    return meta


def resolve_doctor_id(source: Dict[str, Any]) -> str:
    meta = source.get("metadata") if isinstance(source.get("metadata"), dict) else source
    uid = meta.get("U_ID") if isinstance(meta, dict) else None
    return str(uid or "")


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="HLINK board_idx 기준 전 의사 similarity 계산")
    parser.add_argument("--board_idx", type=int, required=True)
    parser.add_argument("--index_name", type=str, default=os.getenv("INDEX_NAME_HL", ""))
    parser.add_argument("--vector_field", type=str, default=os.getenv("HLINK_VECTOR_FIELD", "vector_field"))
    parser.add_argument("--output_csv", type=str, default="data/infer/hlink_similarity.csv")
    args = parser.parse_args()

    if not args.index_name:
        raise RuntimeError("OpenSearch 후보 인덱스명이 필요합니다. --index_name 또는 INDEX_NAME_HL 설정")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # 1) DB에서 공고 텍스트 로드
    conn = get_db_conn()
    try:
        job_text, job_row = fetch_job_text(conn, args.board_idx)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not job_text.strip():
        raise RuntimeError("공고 텍스트가 비어 있습니다. TITLE/CONTENT 확인")

    # 2) 공고 텍스트 임베딩
    job_vec = embed_job_text(job_text)
    u_job = to_unit_vector(job_vec)
    if u_job is None:
        raise RuntimeError("공고 임베딩 정규화 실패")

    # 3) OpenSearch에서 전체 후보 스캔
    os_client = init_opensearch_client()
    hits = scan_all_candidates(os_client, args.index_name, vector_field=args.vector_field, page_size=1000)
    if not hits:
        raise RuntimeError("OpenSearch 후보 인덱스에서 문서를 찾지 못했습니다.")

    # 4) 코사인 유사도 계산
    records: List[Dict[str, Any]] = []
    for h in hits:
        source = h.get("_source", {})
        # 벡터 필드 획득
        vec = None
        try:
            vf = source.get(args.vector_field) if args.vector_field in source else source.get("vector_field")
            if isinstance(vf, dict) and isinstance(vf.get("vector"), list):
                vec = vf.get("vector")
            elif isinstance(vf, list):
                vec = vf
        except Exception:
            vec = None
        if not isinstance(vec, list):
            continue
        u_cand = to_unit_vector(vec)
        if u_cand is None:
            continue
        sim = compute_cosine(u_job, u_cand)

        doc_id = resolve_doctor_id(source)
        meta = extract_meta(source)
        records.append({
            "doctor_id": doc_id,
            "board_id": int(args.board_idx),
            "similarity": float(sim),
            **{f"meta_{k}": v for k, v in meta.items()}
        })

    if not records:
        raise RuntimeError("유사도 계산 결과가 비었습니다 (유효 벡터 없음)")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()


