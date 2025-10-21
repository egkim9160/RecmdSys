#!/usr/bin/env python3
"""
HLINK: board_idx로 04단계 출력 포맷(index, doctor_id, board_id, ... ) 생성

소스:
- Job: medigate.HLINK_JOB (board_idx 단일)
- Doctor: OpenSearch 인덱스(의사 문서 전체, vector 포함)

정책:
- doctor 벡터는 OpenSearch의 vector_field를 사용(의사 임베딩 금지)
- job 텍스트만 임베딩하여 코사인 유사도 계산
- 거리(distance_home/office)는 HOME_ADDR/OFFICE_ADDR 및 ORG_NAME 지오코딩으로 계산(가능 시)
- 경력 요건은 HLINK_JOB에 명시 없음 → "조건 없음" 가정
  (is_career_irrelevant=1, career_match=1, career_gap=0)
- applied=0 (inference용)

출력:
- CSV 컬럼(04단계 결과 동일):
  index, doctor_id, board_id, spec_match, distance_home, distance_office,
  CAREER_YEARS, PAY, career_match, career_gap, is_career_irrelevant,
  similarity, applied
"""

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def clean_html(text_html: str) -> str:
    try:
        from module.html_utils import clean_html_and_get_urls
        txt, _ = clean_html_and_get_urls(text_html)
        return txt
    except Exception:
        return text_html


def geocode(address: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(address, str) or not address.strip():
        return None, None
    try:
        from module.naver_geo import geocode_naver
        return geocode_naver(address.strip())
    except Exception:
        return None, None


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

    # 인증 정보가 모두 있을 때만 http_auth 전달
    if isinstance(username, str) and username and isinstance(password, str) and password:
        kwargs["http_auth"] = (username, password)

    return OpenSearch(**kwargs)


def scan_all_candidates(client, index_name: str, page_size: int = 1000) -> List[Dict[str, Any]]:
    body = {
        "size": page_size,
        "_source": True,
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


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def embed_job_text(job_text: str) -> List[float]:
    from module.llm_utils import get_embedding_client
    client = get_embedding_client()
    if client is None:
        raise RuntimeError("임베딩 클라이언트를 초기화할 수 없습니다. OPENAI_API_KEY/OPENAI_BASE_URL 확인")
    model = os.getenv("HLINK_EMBEDDING_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"))
    resp = client.embeddings.create(input=[job_text], model=model)
    return list(resp.data[0].embedding)


def normalize_specialty(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return "내과" if "내과" in s else s


def fetch_spc_code_names(conn, codes: List[str]) -> Dict[str, str]:
    # CODE_MASTER에서 KBN='SPC' 기준 코드→코드명 매핑 조회
    mapping: Dict[str, str] = {}
    if not codes:
        return mapping
    placeholders = ",".join(["%s"] * len(codes))
    sql = f"SELECT CODE, CODE_NAME FROM CODE_MASTER WHERE KBN='SPC' AND CODE IN ({placeholders})"
    try:
        df = pd.read_sql(sql, conn, params=tuple(codes))
        if df.empty:
            return mapping
        for _, row in df.iterrows():
            code = str(row.get("CODE") or "").strip()
            name = str(row.get("CODE_NAME") or "").strip()
            if code and name:
                mapping[code] = name
        return mapping
    except Exception:
        return mapping


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    import math
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(dlmb / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(r * c)


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="HLINK HLINK_IDX → 04단계 출력 CSV 생성")
    parser.add_argument("--hlink_idx", type=int, required=False, help="HLINK_JOB.HLINK_IDX (권장)")
    parser.add_argument("--board_idx", type=int, required=False, help="과거 호환용(=HLINK_IDX)")
    parser.add_argument("--index_name", type=str, default=os.getenv("INDEX_NAME_HL", ""))
    parser.add_argument("--vector_field", type=str, default=os.getenv("HLINK_VECTOR_FIELD", "vector_field"))
    parser.add_argument("--output_csv", type=str, default="data/processed/training/infer_pairs.csv")
    parser.add_argument("--geocode_workers", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=0, help=">0이면 OpenSearch KNN 상위 N명만 가져옵니다(전체 스캔 방지)")
    parser.add_argument("--no_geocode", action="store_true", help="거리 계산 비활성화(외부 지오코딩 호출 없음)")
    parser.add_argument("--candidates_csv", type=str, default="", help="export_candidates_from_opensearch.py로 생성한 candidates_ok.csv 경로")
    parser.add_argument("--addr_km", type=int, default=30, help="주소 매칭 임계(km). 이하이면 매칭으로 간주")
    args = parser.parse_args()

    if not args.index_name:
        raise RuntimeError("OpenSearch 후보 인덱스명이 필요합니다. --index_name 또는 INDEX_NAME_HL 설정")

    # 파라미터 정규화: HLINK_IDX 우선
    if args.hlink_idx is not None:
        target_hlink_idx = int(args.hlink_idx)
    elif args.board_idx is not None:
        target_hlink_idx = int(args.board_idx)
    else:
        raise RuntimeError("--hlink_idx(권장) 또는 --board_idx 중 하나를 지정하세요.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # 1) Job 텍스트/메타
    conn = get_db_conn()
    try:
        sql = (
            "SELECT J.HLINK_IDX, J.TITLE, J.CONTENT, J.ORG_NAME, J.PAY, J.PAY_TYPE, J.SPC_CODE, J.SPC_DETAIL_CODE, "
            "       J.CLIENT_IDX, C.ADDR "
            "FROM medigate.HLINK_JOB J "
            "LEFT JOIN medigate.HLINK_CLIENT C ON C.CLIENT_IDX = J.CLIENT_IDX "
            "WHERE J.HLINK_IDX = %s AND J.DISPLAY_FLAG='Y' "
            "AND (J.DEL_FLAG IS NULL OR J.DEL_FLAG <> 'Y') LIMIT 1"
        )
        job_df = pd.read_sql(sql, conn, params=(target_hlink_idx,))
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if job_df.empty:
        raise RuntimeError(f"HLINK_IDX={target_hlink_idx} 공고를 찾지 못했습니다.")
    j = job_df.iloc[0].to_dict()
    title = (j.get("TITLE") or "").strip()
    content_clean = clean_html(str(j.get("CONTENT") or ""))
    job_text = "\n\n".join([t for t in [title, content_clean] if t])
    # 공고 전문과 코드 → 코드명 매핑 (간단 SQL)
    try:
        conn2 = get_db_conn()
        sql_map = (
            "SELECT GROUP_CONCAT(cm.CODE_NAME ORDER BY cm.CODE_NAME SEPARATOR ', ') AS SPECIALTIES_MAPPED "
            "FROM medigate.HLINK_JOB j "
            "JOIN medigate.CODE_MASTER cm ON cm.KBN='SPC' "
            " AND (FIND_IN_SET(cm.CODE, j.SPC_CODE) > 0 OR FIND_IN_SET(cm.CODE, j.SPC_DETAIL_CODE) > 0) "
            "WHERE j.HLINK_IDX = %s"
        )
        df_map = pd.read_sql(sql_map, conn2, params=(target_hlink_idx,))
        mapped_str = str(df_map.iloc[0]["SPECIALTIES_MAPPED"]).strip() if (not df_map.empty and df_map.iloc[0]["SPECIALTIES_MAPPED"] is not None) else ""
        job_spec_tokens = [t.strip() for t in mapped_str.split(',') if t.strip()]
    except Exception:
        job_spec_tokens = []
    finally:
        try:
            conn2.close()
        except Exception:
            pass

    # 지오코딩: ADDR 우선, 없으면 ORG_NAME
    org_name = str(j.get("ORG_NAME") or "").strip()
    addr = str(j.get("ADDR") or "").strip()
    addr_for_geo = addr if addr else org_name
    org_lat, org_lon = geocode(addr_for_geo)

    # PAY: HLINK 원시값 그대로 사용 (모델 호환 위해 float 변환, None→0)
    try:
        pay_val = float(j.get("PAY")) if j.get("PAY") is not None and str(j.get("PAY")).strip() != "" else 0.0
    except Exception:
        pay_val = 0.0

    # LLM으로 CONTENT에서 경력 요구 파싱 → "경력 무관" 또는 숫자
    def extract_career_req(text: str) -> Tuple[str, Optional[int]]:
        try:
            from module.llm_utils import get_openai_client
            cli = get_openai_client()
            if cli is None:
                return ("경력 무관", None)
            prompt_sys = (
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
            prompt_user = f"다음 공고 본문에서 경력 요구를 추출해. 본문:\n{text}\n정답:"
            resp = cli.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user},
                ],
            )
            out = resp.choices[0].message.content.strip()
            if "무관" in out:
                return ("경력 무관", None)
            try:
                num = int(float(out))
                return (str(num), num)
            except Exception:
                return ("경력 무관", None)
        except Exception:
            return ("경력 무관", None)

    career_text, career_num = extract_career_req(content_clean)

    # 2) Job 텍스트 임베딩 → 단위벡터
    job_vec = embed_job_text(job_text)
    u_job = to_unit_vector(job_vec)
    if u_job is None:
        raise RuntimeError("공고 임베딩 정규화 실패")

    # Board 메타 저장 (output_csv와 같은 디렉터리)
    try:
        board_meta = {
            "HLINK_IDX": int(target_hlink_idx),
            "TITLE": j.get("TITLE"),
            "ORG_NAME": org_name,
            "CLIENT_IDX": j.get("CLIENT_IDX"),
            "ADDR": addr or None,
            "ORG_lat": float(org_lat) if org_lat is not None else None,
            "ORG_lon": float(org_lon) if org_lon is not None else None,
            "PAY": float(pay_val),
            "PAY_TYPE": j.get("PAY_TYPE"),
            "SPC_CODE": j.get("SPC_CODE"),
            "SPC_DETAIL_CODE": j.get("SPC_DETAIL_CODE"),
            # embedding inputs
            "EMBED_TITLE": title,
            "EMBED_CONTENT_CLEAN": content_clean,
            # mapped specialties
            "SPECIALTIES_MAPPED": ",".join(job_spec_tokens) if isinstance(job_spec_tokens, list) else None,
            # LLM parsed career
            "CAREER_TEXT": career_text,
            "CAREER_NUM": int(career_num) if isinstance(career_num, int) else None,
        }
        out_dir_for_board = os.path.dirname(os.path.abspath(args.output_csv))
        os.makedirs(out_dir_for_board, exist_ok=True)
        pd.DataFrame([board_meta]).to_csv(os.path.join(out_dir_for_board, "board_meta.csv"), index=False)
    except Exception:
        pass

    # 3) 후보 수집: (A) --candidates_csv 제공 시 로컬 CSV 사용, (B) 없으면 OpenSearch 사용
    candidates: List[Dict[str, Any]] = []
    if args.candidates_csv and os.path.exists(args.candidates_csv):
        df_c = pd.read_csv(args.candidates_csv)
        # 필요한 최소 컬럼 확인
        for col in ["U_ID","SPECIALTY","CAREER_YEARS","geo_ADDR_lat","geo_ADDR_lon","geo_OFFICE_ADDR_lat","geo_OFFICE_ADDR_lon","vector_field"]:
            if col not in df_c.columns:
                raise RuntimeError(f"candidates_csv에 필수 컬럼 누락: {col}")
        # 로컬 CSV → candidates 리스트 변환
        for _, r in df_c.iterrows():
            try:
                vec = json.loads(r.get("vector_field")) if isinstance(r.get("vector_field"), str) else None
            except Exception:
                vec = None
            candidates.append({
                "U_ID": str(r.get("U_ID")),
                "SPECIALTY": r.get("SPECIALTY"),
                "HOME_ADDR": None,  # 주소 텍스트는 불필요
                "OFFICE_ADDR": None,
                "CAREER_YEARS": r.get("CAREER_YEARS"),
                "geo_ADDR_lat": r.get("geo_ADDR_lat"),
                "geo_ADDR_lon": r.get("geo_ADDR_lon"),
                "geo_OFFICE_ADDR_lat": r.get("geo_OFFICE_ADDR_lat"),
                "geo_OFFICE_ADDR_lon": r.get("geo_OFFICE_ADDR_lon"),
                "vec": vec,
            })
    else:
        os_client = init_opensearch_client()
        def _knn_candidates(client, index_name: str, vector_field: str, vec: List[float], size: int) -> List[Dict[str, Any]]:
            body = {
                "size": int(size),
                "_source": True,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    vector_field: {
                                        "vector": vec,
                                        "k": int(size)
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            res = client.search(index=index_name, body=body)
            return res.get("hits", {}).get("hits", [])

        if int(args.top_k) > 0:
            hits = _knn_candidates(os_client, args.index_name, args.vector_field, job_vec, int(args.top_k))
        else:
            hits = scan_all_candidates(os_client, args.index_name, page_size=1000)
        if not hits:
            raise RuntimeError("OpenSearch 인덱스에서 후보 문서를 찾지 못했습니다.")

        def _get_meta(source: Dict[str, Any]) -> Dict[str, Any]:
            meta = source.get("metadata") if isinstance(source.get("metadata"), dict) else source
            return meta if isinstance(meta, dict) else {}

        home_addrs: Dict[int, str] = {}
        office_addrs: Dict[int, str] = {}
        for i, h in enumerate(hits):
            s = h.get("_source", {})
            meta = _get_meta(s)
            uid = str(meta.get("U_ID") or "").strip()
            if not uid:
                continue
            vf = s.get(args.vector_field) if args.vector_field in s else s.get("vector_field")
            vec = None
            if isinstance(vf, dict) and isinstance(vf.get("vector"), list):
                vec = vf.get("vector")
            elif isinstance(vf, list):
                vec = vf
            if not isinstance(vec, list):
                continue
            candidates.append({
                "U_ID": uid,
                "SPECIALTY": meta.get("SPECIALTY"),
                "HOME_ADDR": meta.get("HOME_ADDR"),
                "OFFICE_ADDR": meta.get("OFFICE_ADDR"),
                "CAREER_YEARS": meta.get("CAREER_YEARS"),
                "vec": vec,
            })

    # 4) 주소 수집 및 지오코딩(병렬) — no_geocode이면 건너뜀, 또는 candidates_csv 기반이면 이미 좌표 존재

    home_addrs: Dict[int, str] = {}
    office_addrs: Dict[int, str] = {}
    if not args.candidates_csv:
        for i, c in enumerate(candidates):
            if not args.no_geocode:
                if isinstance(c.get("HOME_ADDR"), str) and c.get("HOME_ADDR").strip():
                    home_addrs[i] = c.get("HOME_ADDR").strip()
                if isinstance(c.get("OFFICE_ADDR"), str) and c.get("OFFICE_ADDR").strip():
                    office_addrs[i] = c.get("OFFICE_ADDR").strip()

    def _batch_geocode(addrs: List[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        if not addrs:
            return {}
        results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        def task(a: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
            if a in cache:
                return a, cache[a]
            lat, lon = geocode(a)
            cache[a] = (lat, lon)
            return a, (lat, lon)
        with ThreadPoolExecutor(max_workers=max(1, int(args.geocode_workers))) as ex:
            futs = [ex.submit(task, a) for a in sorted(set(addrs))]
            for f in as_completed(futs):
                a, coord = f.result()
                results[a] = coord
        return results
    if args.candidates_csv:
        home_coords = {}
        office_coords = {}
    else:
        if args.no_geocode:
            home_coords = {}
            office_coords = {}
        else:
            home_coords = _batch_geocode(list(home_addrs.values()))
            office_coords = _batch_geocode(list(office_addrs.values()))

    # 5) 레코드 생성(04 단계 스키마)
    records: List[Dict[str, Any]] = []
    job_specs_norm = set([normalize_specialty(t) for t in job_spec_tokens if normalize_specialty(t)]) if job_spec_tokens else set()

    for c in candidates:
        uid = c["U_ID"]
        spec = normalize_specialty(c.get("SPECIALTY"))
        u_vec = to_unit_vector(c.get("vec") or [])
        if u_vec is None:
            continue
        sim = cosine(u_job, u_vec)
        # 거리
        d_home = d_office = None
        if args.candidates_csv:
            # 좌표가 CSV에 포함되어 있으므로 바로 사용
            h_lat, h_lon = c.get("geo_ADDR_lat"), c.get("geo_ADDR_lon")
            o_lat, o_lon = c.get("geo_OFFICE_ADDR_lat"), c.get("geo_OFFICE_ADDR_lon")
            if isinstance(h_lat, float) and isinstance(h_lon, float) and org_lat is not None and org_lon is not None:
                d_home = int(round(haversine_km(h_lat, h_lon, org_lat, org_lon)))
            if isinstance(o_lat, float) and isinstance(o_lon, float) and org_lat is not None and org_lon is not None:
                d_office = int(round(haversine_km(o_lat, o_lon, org_lat, org_lon)))
        elif not args.no_geocode and org_lat is not None and org_lon is not None:
            h_addr = c.get("HOME_ADDR")
            o_addr = c.get("OFFICE_ADDR")
            h_lat, h_lon = home_coords.get(h_addr, (None, None)) if isinstance(h_addr, str) else (None, None)
            o_lat, o_lon = office_coords.get(o_addr, (None, None)) if isinstance(o_addr, str) else (None, None)
            if h_lat is not None and h_lon is not None:
                d_home = int(round(haversine_km(h_lat, h_lon, org_lat, org_lon)))
            if o_lat is not None and o_lon is not None:
                d_office = int(round(haversine_km(o_lat, o_lon, org_lat, org_lon)))

        # spec_match
        spec_match = 1 if (spec and spec in job_specs_norm) else 0

        # career years
        cy = c.get("CAREER_YEARS")
        try:
            c_years = int(cy) if cy is not None and str(cy).strip() != "" else 0
        except Exception:
            # 텍스트 규칙 최소화: 'N년' 패턴만 처리
            try:
                import re
                m = re.search(r"(\d+)", str(cy))
                c_years = int(m.group(1)) if m else 0
            except Exception:
                c_years = 0

        # career 요건 없음 가정
        career_match = 1
        career_gap = 0
        is_career_irrelevant = 1

        # 거리 NaN 방지: 없으면 -1로 표기(학습셋과 다를 수 있으니 주의)
        d_home = int(d_home) if d_home is not None else -1
        d_office = int(d_office) if d_office is not None else -1

        records.append({
            "doctor_id": str(uid),
            "board_id": int(target_hlink_idx),
            "spec_match": int(spec_match),
            "distance_home": int(d_home),
            "distance_office": int(d_office),
            "CAREER_YEARS": int(c_years),
            "PAY": float(pay_val),
            "ORG_lat": float(org_lat) if org_lat is not None else None,
            "ORG_lon": float(org_lon) if org_lon is not None else None,
            "career_match": int(career_match),
            "career_gap": int(career_gap),
            "is_career_irrelevant": int(is_career_irrelevant),
            "similarity": float(sim),
            "applied": 0,
        })

    if not records:
        raise RuntimeError("생성된 04단계 입력 레코드가 없습니다 (유효 후보 없음)")

    df = pd.DataFrame.from_records(records)
    df = df.reset_index(drop=True)
    df.insert(0, "index", df.index.astype(int))
    cols = [
        "index","doctor_id","board_id","spec_match","distance_home","distance_office",
        "CAREER_YEARS","PAY","ORG_lat","ORG_lon","career_match","career_gap","is_career_irrelevant","similarity","applied"
    ]
    df = df[cols]
    df.to_csv(args.output_csv, index=False)
    print(args.output_csv)


if __name__ == "__main__":
    main()


