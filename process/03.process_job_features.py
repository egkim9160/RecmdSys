#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job(공고) raw CSV → processed view 단일 스크립트

포함 기능:
- 급여 변환(원 단위): Gross 연봉↔Net 월급 계산 로직 내장
- 공고 뷰 생성: PAY 산출, 지오코딩(ADDRESS → LLM 정제 → REGION), 로그/노트 출력

입출력:
- 입력: /SPO/Project/RecSys/data/raw/job_features.csv (기본)
- 출력: /SPO/Project/RecSys/data/processed/job_features_{FIXED_TS}/job_training_view.csv

사용 예:
python process_job_features.py --input /path/to/job_features.csv --out_dir /path/to/processed_dir --limit 1000 --no-llm --concurrency 20 --verbose
"""

import os
import sys
import argparse
import asyncio
import time
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
from dotenv import load_dotenv
import re


# -----------------------------
# 고정 타임스탬프 및 경로
# -----------------------------
FIXED_TS = "20250925_134845"
DEFAULT_OUT_DIR = f"/SPO/Project/RecSys/data/processed/job_features_{FIXED_TS}"


# -----------------------------
# 환경 로딩 / 외부 유틸
# -----------------------------
from pathlib import Path
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


def load_environment() -> None:
    # Always load from project root (run_pipeline.py location)
    try:
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        if os.path.exists(".env"):
            load_dotenv(".env")


def get_openai_client():
    from module.llm_utils import get_openai_client as get_client
    return get_client()


def get_embedding_client():
    from module.llm_utils import get_embedding_client as get_embed_client
    return get_embed_client()


def _batch_embed_texts(client, texts, model: str = "text-embedding-3-large", batch_size: int = 64):
    from module.llm_utils import batch_embed_texts
    return batch_embed_texts(client, texts, model, batch_size)


def clean_address_with_llm(raw_address: str, client) -> Optional[str]:
    from module.llm_utils import clean_address_with_llm as clean_addr
    return clean_addr(raw_address, client)


def try_geocode(address: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(address, str):
        return None, None
    a = address.strip()
    if not a:
        return None, None
    try:
        lat, lon = geocode_naver(a)
        return lat, lon
    except Exception:
        return None, None


# -----------------------------
# 급여 변환 로직 (원 단위 I/O) - 내장
# -----------------------------
# 종합소득세 누진표 (Quick Calculation) (하한, 상한, 한계세율, 누진공제용 base_tax)
PIT_BRACKETS = [
    (0,             14_000_000, 0.06, 0),
    (14_000_000,    50_000_000, 0.15, 840_000),
    (50_000_000,    88_000_000, 0.24, 6_240_000),
    (88_000_000,   150_000_000, 0.35, 15_360_000),
    (150_000_000,  300_000_000, 0.38, 37_060_000),
    (300_000_000,  500_000_000, 0.40, 94_060_000),
    (500_000_000,1_000_000_000, 0.42, 174_060_000),
    (1_000_000_000,        float('inf'), 0.45, 384_060_000),
]
LOCAL_TAX_RATE = 0.10  # 지방소득세(국세 세액공제 후)의 10%

# 4대보험(근로자 부담)
NPS_RATE_EMP = 0.045               # 국민연금 4.5%
NPS_MONTHLY_BASE_CAP = 6_370_000   # 국민연금 월 기준소득 상한
HI_RATE_EMP = 0.0709 / 2           # 건강보험 근로자 3.545%
LTC_RATE_ON_HI = 0.1295            # 장기요양 = 건보료의 12.95%
EI_RATE_EMP = 0.009                # 고용보험 근로자 0.9%

# 인적공제/근로소득공제 상수
BASIC_DED_PER_HEAD = 1_500_000     # 1인당 150만원
EMPLOYMENT_DED_CAP = 20_000_000    # 연 2천만원


def _employment_income_deduction(annual_gross: int) -> int:
    g = annual_gross
    if g <= 5_000_000:
        d = 0.70 * g
    elif g <= 15_000_000:
        d = 3_500_000 + 0.40 * (g - 5_000_000)
    elif g <= 45_000_000:
        d = 7_500_000 + 0.15 * (g - 15_000_000)
    elif g <= 100_000_000:
        d = 12_000_000 + 0.05 * (g - 45_000_000)
    else:
        d = 14_750_000 + 0.02 * (g - 100_000_000)
    return int(min(d, EMPLOYMENT_DED_CAP))


def _basic_personal_deduction(headcount: int = 1) -> int:
    return BASIC_DED_PER_HEAD * max(0, int(headcount))


def _monthly_social_insurance(monthly_gross: float) -> Tuple[float, float, float, float]:
    nps_base = min(monthly_gross, NPS_MONTHLY_BASE_CAP)
    nps = nps_base * NPS_RATE_EMP
    hi = monthly_gross * HI_RATE_EMP
    ltc = hi * LTC_RATE_ON_HI
    ei = monthly_gross * EI_RATE_EMP
    return nps, hi, ltc, ei


def _annual_social_insurance(annual_gross: int) -> float:
    m = annual_gross / 12.0
    nps, hi, ltc, ei = _monthly_social_insurance(m)
    return (nps + hi + ltc + ei) * 12.0


def _compute_pit_before_credit(taxable_income: float) -> float:
    ti = max(0.0, taxable_income)
    for lower, upper, rate, base_tax in PIT_BRACKETS:
        if ti <= upper:
            if lower == 0:
                return ti * rate
            return base_tax + rate * (ti - lower)
    return 0.0


def _earned_income_tax_credit(pit_before_credit: float, annual_gross: int) -> float:
    if pit_before_credit <= 1_300_000:
        prelim = 0.55 * pit_before_credit
    else:
        prelim = 715_000 + 0.30 * (pit_before_credit - 1_300_000)

    g = annual_gross
    if g <= 33_000_000:
        cap = 740_000
    elif g <= 70_000_000:
        cap = max(740_000 - (g - 33_000_000) * 0.008, 660_000)
    elif g <= 120_000_000:
        cap = max(660_000 - (g - 70_000_000) * 0.5, 500_000)
    else:
        cap = max(500_000 - (g - 120_000_000) * 0.5, 200_000)

    return max(0.0, min(prelim, cap))


def _gross_to_net_detail(annual_gross: int, headcount: int = 1) -> Dict[str, float]:
    emp_ded = _employment_income_deduction(annual_gross)
    pers_ded = _basic_personal_deduction(headcount)
    soc = _annual_social_insurance(annual_gross)
    taxable = max(0.0, annual_gross - emp_ded - pers_ded - soc)

    pit_before = _compute_pit_before_credit(taxable)
    credit = _earned_income_tax_credit(pit_before, annual_gross)
    pit_after = max(0.0, pit_before - credit)

    local_tax = pit_after * LOCAL_TAX_RATE
    total_tax = pit_after + local_tax
    annual_net = annual_gross - soc - total_tax

    return {
        "annual_gross": float(annual_gross),
        "annual_net": float(annual_net),
        "monthly_net": float(annual_net / 12.0),
    }


def gross_annual_to_net_monthly_won(gross_annual_won: float, headcount: int = 1) -> int:
    res = _gross_to_net_detail(int(gross_annual_won), headcount=headcount)
    return int(round(res["monthly_net"]))


def net_monthly_to_gross_annual_won(target_net_monthly_won: float, headcount: int = 1, tol_won: int = 1_000) -> int:
    target_annual_net_won = float(target_net_monthly_won) * 12.0
    lo = target_annual_net_won
    hi = target_annual_net_won * 2.8 + 30_000_000.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        annual_net = _gross_to_net_detail(int(mid), headcount=headcount)["annual_net"]
        if annual_net < target_annual_net_won:
            lo = mid
        else:
            hi = mid
        if abs(annual_net - target_annual_net_won) < tol_won:
            break
    return int(round(hi))


# -----------------------------
# PAY 계산/지오코딩
# -----------------------------

def to_float_or_none(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def compute_monthly_pay(row: pd.Series, llm_client) -> Optional[float]:
    ptype = str(row.get("PAY_TYPE_NAME", ""))
    norm = ptype.replace(" ", "")

    if "Net(세후)월급" in norm:
        return to_float_or_none(row.get("PAY_MONTH"))

    if "Gross(세전)연봉" in norm:
        net_m = to_float_or_none(row.get("CONVERTED_NET_MONTHLY_WON"))
        if net_m is not None:
            return net_m
        gross_annual = to_float_or_none(row.get("PAY_YEAR"))
        if gross_annual is None:
            return None
        try:
            return float(gross_annual_to_net_monthly_won(gross_annual))
        except Exception:
            return None

    if "Day일급" in norm or "Day일급" == norm or "Day일급" in norm:
        day_pay = to_float_or_none(row.get("PAY_DAY"))
        if day_pay is None:
            return None
        return day_pay * 20.0

    m = to_float_or_none(row.get("PAY_MONTH"))
    if m is not None:
        return m
    return None


async def ageocode_with_fallback(address: Optional[str], region: Optional[str], llm_client, *, sem: asyncio.Semaphore, geo_cache: Dict[str, Tuple[Optional[float], Optional[float]]], clean_cache: Dict[str, Optional[str]], lock: asyncio.Lock, geocode_logs: List[Dict[str, Any]], llm_logs: List[Dict[str, Any]], board_idx: Optional[Any]) -> Tuple[Optional[float], Optional[float]]:
    async def cached_geocode(key: Optional[str], stage: str) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(key, str) or not key.strip():
            return None, None
        k = key.strip()
        async with lock:
            if k in geo_cache:
                return geo_cache[k]
        async with sem:
            lat, lon = await asyncio.to_thread(lambda: try_geocode(k))
        async with lock:
            # stage별 로그는 간략화 (필요 시 확장)
            geocode_logs.append({"stage": stage, "board_idx": board_idx, "address": k, "lat": lat, "lon": lon})
            geo_cache[k] = (lat, lon)
        return lat, lon

    # 1) 주소 우선
    lat, lon = await cached_geocode(address, "address")
    if lat is not None and lon is not None:
        return lat, lon

    # 2) LLM 정제 후 재시도
    cleaned: Optional[str] = None
    if isinstance(address, str) and address.strip():
        key = address.strip()
        async with lock:
            if key in clean_cache:
                cleaned = clean_cache[key]
        if cleaned is None:
            async with sem:
                cleaned = await asyncio.to_thread(lambda: clean_address_with_llm(address, llm_client))
            async with lock:
                llm_logs.append({"board_idx": board_idx, "raw": address, "cleaned": cleaned})
                clean_cache[key] = cleaned
        if cleaned:
            lat, lon = await cached_geocode(cleaned, "cleaned")
            if lat is not None and lon is not None:
                return lat, lon

    # 3) REGION 시도
    lat, lon = await cached_geocode(region if isinstance(region, str) else None, "region")
    if lat is not None and lon is not None:
        return lat, lon

    return None, None


# -----------------------------
# CLI / Main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Job raw CSV → processed view (급여변환+지오코딩)")
    p.add_argument("--input", default="/SPO/Project/RecSys/data/raw/job_features.csv", help="입력 job_features CSV 경로")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="출력 디렉터리")
    p.add_argument("--limit", type=int, default=None, help="처리 행 수 제한(옵션)")
    p.add_argument("--no-llm", action="store_true", help="LLM 주소 정제 비활성화")
    p.add_argument("--concurrency", type=int, default=20, help="지오코딩 동시 실행 한도")
    p.add_argument("--verbose", action="store_true", help="진행 로그 출력")
    p.add_argument("--log-interval", type=int, default=200, help="진행 로그 출력 간격(건수)")
    p.add_argument("--embed_model", type=str, default="text-embedding-3-large", help="임베딩 모델명(OpenAI 호환)")
    p.add_argument("--embed_batch", type=int, default=64, help="임베딩 배치 크기")
    p.add_argument("--no-embed", action="store_true", help="임베딩 비활성화")
    return p


def main() -> None:
    load_environment()

    parser = build_parser()
    args = parser.parse_args()

    input_csv = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    # LLM 클라이언트 준비
    llm_client = None if args.no_llm else get_openai_client()

    # PAY 계산
    pays = []
    total = len(df)
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows(), 1):
        pays.append(compute_monthly_pay(row, llm_client))
        if args.verbose and (i % max(1, args.log_interval) == 0 or i == total):
            print(f"[PAY] progress {i}/{total} elapsed={time.time() - t0:.1f}s")

    # 지오코딩 (ADDRESS -> LLM 정제 -> REGION) - asyncio 비동기 처리
    async def run_geocoding(rows: List[pd.Series]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        sem = asyncio.Semaphore(max(1, args.concurrency))
        geo_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        clean_cache: Dict[str, Optional[str]] = {}
        lock = asyncio.Lock()
        geocode_logs: List[Dict[str, Any]] = []
        llm_logs: List[Dict[str, Any]] = []

        async def task(i: int, row: pd.Series) -> Tuple[int, Optional[float], Optional[float]]:
            lat, lon = await ageocode_with_fallback(
                row.get("ADDRESS"),
                row.get("REGION"),
                llm_client,
                sem=sem,
                geo_cache=geo_cache,
                clean_cache=clean_cache,
                lock=lock,
                geocode_logs=geocode_logs,
                llm_logs=llm_logs,
                board_idx=row.get("BOARD_IDX"),
            )
            return i, lat, lon

        if args.verbose:
            print(f"[GEO] Start geocoding: total={len(df)}, concurrency={args.concurrency}, llm={'off' if args.no_llm else 'on'}")
        start_ts = time.time()
        tasks = [asyncio.create_task(task(i, df.iloc[i])) for i in range(len(df))]
        results: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * len(df)
        done = ok = fail = 0
        for fut in asyncio.as_completed(tasks):
            i, lat, lon = await fut
            results[i] = (lat, lon)
            done += 1
            if lat is not None and lon is not None:
                ok += 1
            else:
                fail += 1
            if args.verbose and (done % max(1, args.log_interval) == 0 or done == len(df)):
                elapsed = time.time() - start_ts
                print(f"[GEO] progress {done}/{len(df)} ok={ok} fail={fail} elapsed={elapsed:.1f}s")

        lats = [lat for lat, _ in results]
        lons = [lon for _, lon in results]
        return lats, lons, geocode_logs, llm_logs

    org_lats, org_lons, geocode_logs, llm_logs = asyncio.run(run_geocoding([df.iloc[i] for i in range(len(df))]))

    # -----------------------------
    # 경력 요구사항 파싱/숫자화
    # -----------------------------
    def _parse_job_career_req(desc: Optional[str]) -> Tuple[Optional[int], Optional[int], int]:
        """공고의 경력 요구사항 텍스트를 (min_years, max_years, is_irrelevant)로 변환.
        규칙은 사용자 경력 정규화와 유사한 lower-bound 중심 처리.
        - "경력무관" → (None, None, 1)
        - "신입"/"신입가능" → (0, 1, 0)
        - "N~M년" → (N, M, 0)
        - "N년 이상" → (N, None, 0)
        - 인식불가/없음 → (None, None, 1)  # 조건 없음으로 간주
        """
        try:
            s = str(desc) if desc is not None else ""
        except Exception:
            s = ""
        s = s.replace(" ", "").strip()
        if not s:
            return None, None, 1
        if ("경력무관" in s) or ("무관" in s):
            return None, None, 1
        if ("신입" in s):
            return 0, 1, 0
        m = re.match(r"^(\d+)[~\-∼](\d+)년$", s)
        if m:
            try:
                mn = int(m.group(1))
                mx = int(m.group(2))
                return mn, mx, 0
            except Exception:
                return None, None, 1
        m = re.match(r"^(\d+)년이상$", s)
        if m:
            try:
                mn = int(m.group(1))
                return mn, None, 0
            except Exception:
                return None, None, 1
        m = re.match(r"^(\d+)년$", s)
        if m:
            try:
                mn = int(m.group(1))
                return mn, mn, 0
            except Exception:
                return None, None, 1
        return None, None, 1

    # 입력 DF에서 경력 요구사항 컬럼이 있는 경우 파싱하여 숫자 컬럼 생성
    career_desc_series = df.get("CAREER_REQ_DESC")
    job_min_career_years: List[Optional[int]] = []
    job_max_career_years: List[Optional[int]] = []
    job_is_career_irrelevant: List[int] = []
    if career_desc_series is not None:
        for v in career_desc_series:
            mn, mx, irr = _parse_job_career_req(v)
            job_min_career_years.append(mn)
            job_max_career_years.append(mx)
            job_is_career_irrelevant.append(int(irr))
    else:
        # 컬럼이 없으면 조건 없음으로 처리
        job_min_career_years = [None] * len(df)
        job_max_career_years = [None] * len(df)
        job_is_career_irrelevant = [1] * len(df)

    # 출력 DF 구성
    # TITLE + CONTENT 임베딩
    job_texts = []
    job_indices = []
    if not args.no_embed:
        for i in range(len(df)):
            title = str(df.iloc[i].get("TITLE", "") or "").strip()
            content = str(df.iloc[i].get("CONTENT", "") or "").strip()
            t = (title + "\n\n" + content).strip()
            if t:
                job_texts.append(t)
                job_indices.append(i)
    emb_client = None if args.no_embed else get_embedding_client()
    if args.verbose:
        print(f"[EMBED] client={'on' if emb_client is not None else 'off'} model={args.embed_model} batch={args.embed_batch}")
    emb_vecs = _batch_embed_texts(emb_client, job_texts, model=args.embed_model, batch_size=max(1, int(args.embed_batch))) if not args.no_embed else []

    df["JOB_EMB_4096"] = pd.NA
    for ridx, vec in zip(job_indices, emb_vecs):
        if vec is not None:
            try:
                df.at[ridx, "JOB_EMB_4096"] = json.dumps(vec, ensure_ascii=False)
            except Exception:
                df.at[ridx, "JOB_EMB_4096"] = pd.NA

    out_df = pd.DataFrame({
        "BOARD_IDX": df.get("BOARD_IDX"),
        "SPECIALTIES": df.get("SPECIALTIES"),
        "PAY": pays,
        "ORG_lat": org_lats,
        "ORG_lon": org_lons,
        "JOB_EMB_4096": df.get("JOB_EMB_4096"),
        # 경력 요구사항 파생 특성들(가능 시)
        "CAREER_REQ_DESC": career_desc_series if career_desc_series is not None else None,
        "JOB_MIN_CAREER_YEARS": job_min_career_years,
        "JOB_MAX_CAREER_YEARS": job_max_career_years,
        "JOB_IS_CAREER_IRRELEVANT": job_is_career_irrelevant,
    })

    out_csv = os.path.join(out_dir, "job_training_view.csv")
    out_df.to_csv(out_csv, index=False)

    # 노트 파일
    notes_path = os.path.join(out_dir, "NOTES.txt")
    notes = []
    notes.append("입력 파일:\n")
    notes.append(f"- {input_csv}\n\n")
    notes.append("처리 규칙:\n")
    notes.append("- PAY 월급 기준: Net=PAY_MONTH, Gross=CONVERTED_NET_MONTHLY_WON(없으면 즉시계산), Day=PAY_DAY*20\n")
    notes.append("- 지오코딩: ADDRESS -> (실패시) LLM 정제 -> (실패시) REGION -> (실패시) 빈칸\n\n")
    notes.append("출력 컬럼:\n")
    notes.append("- BOARD_IDX, SPECIALTIES, PAY, ORG_lat, ORG_lon\n")
    notes.append("- (옵션) CAREER_REQ_DESC, JOB_MIN_CAREER_YEARS, JOB_MAX_CAREER_YEARS, JOB_IS_CAREER_IRRELEVANT\n\n")
    notes.append(f"타임스탬프(고정): {FIXED_TS}\n")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("".join(notes))

    # 로그 파일 저장 (JSON Lines)
    geo_log_path = os.path.join(out_dir, "geocoding_logs.jsonl")
    llm_log_path = os.path.join(out_dir, "llm_logs.jsonl")
    def _to_jsonable(obj: Any) -> Any:
        try:
            import numpy as _np
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
        except Exception:
            pass
        return obj

    with open(geo_log_path, "w", encoding="utf-8") as f:
        for rec in geocode_logs:
            rec2 = {k: _to_jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
    with open(llm_log_path, "w", encoding="utf-8") as f:
        for rec in llm_logs:
            rec2 = {k: _to_jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n")

    print(f"Saved: {out_csv}")
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()
