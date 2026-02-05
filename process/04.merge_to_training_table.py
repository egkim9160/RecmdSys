import argparse
import os
from datetime import datetime
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import re
import json


def normalize_specialty_label(raw_label: Optional[str]) -> Optional[str]:
    """Normalize specialty label according to domain rules.

    Current rule: If the label contains '내과' as a substring (e.g., '내분비내과', '류마티스내과',
    '소화기내과', '순환기내과'), map it to the umbrella label '내과'. Otherwise, return as-is.
    """
    if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
        return None
    label = str(raw_label).strip()
    if "내과" in label:
        return "내과"
    return label


def parse_job_specialties_to_set(raw_specialties: Optional[str]) -> Optional[Set[str]]:
    """Parse job specialties string into a normalized set of labels.

    The input is a comma-separated string. Each token is stripped and normalized
    using the same rule as user specialty (umbrella mapping for '내과').
    Returns None when parsing is not possible.
    """
    if raw_specialties is None or (isinstance(raw_specialties, float) and np.isnan(raw_specialties)):
        return None
    tokens = [t.strip() for t in str(raw_specialties).split(",") if t.strip()]
    if not tokens:
        return None
    normalized = [normalize_specialty_label(t) for t in tokens]
    return set([t for t in normalized if t is not None])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in kilometers between two lat/lon points."""
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def compute_distances_for_board(
    home_lat: float,
    home_lon: float,
    office_lat: Optional[float],
    office_lon: Optional[float],
    org_lat: float,
    org_lon: float,
) -> Tuple[int, float]:
    """Compute integer km distances from home and office to org.

    Returns:
        (distance_home, distance_office): distance_office is np.nan if office coords are missing
    """
    d_home = haversine_km(home_lat, home_lon, org_lat, org_lon)

    # 직장 좌표가 없으면 NaN 반환 (XGBoost가 자동으로 처리)
    if office_lat is None or office_lon is None or pd.isna(office_lat) or pd.isna(office_lon):
        d_office = np.nan
    else:
        d_office = haversine_km(office_lat, office_lon, org_lat, org_lon)

    return int(round(d_home)), d_office if pd.isna(d_office) else int(round(d_office))


def parse_user_applied_board_ids(raw_board_idx: Optional[str]) -> Set[int]:
    """Parse user's applied board ids string separated by '|'."""
    if raw_board_idx is None or (isinstance(raw_board_idx, float) and np.isnan(raw_board_idx)):
        return set()
    parts = [p.strip() for p in str(raw_board_idx).split("|") if p.strip()]
    ids: Set[int] = set()
    for p in parts:
        try:
            ids.add(int(p))
        except Exception:
            continue
    return ids


def is_heavy_user(
    total_applies: int,
    total_jobs: int,
    max_monthly: Optional[int] = None,
    max_weekly: Optional[int] = None,
) -> bool:
    """
    과다지원자(outlier) 판별 함수

    조건 (OR 연결):
    - 조건 1: 누적 과다지원 (total_applies > 80 AND apply_rate > 0.3%)
    - 조건 2: 월 폭주 (max_monthly > 60)
    - 조건 3: 주 폭주 (max_weekly > 30)

    Args:
        total_applies: 총 지원 횟수
        total_jobs: 전체 공고 수 (apply_rate 계산용)
        max_monthly: 월별 최대 지원 횟수 (선택)
        max_weekly: 주별 최대 지원 횟수 (선택)

    Returns:
        True if heavy user (should be excluded), False otherwise
    """
    # 조건 1: 누적 과다지원 (AND)
    if total_jobs > 0:
        apply_rate = (total_applies / total_jobs) * 100  # percentage
        if total_applies > 80 and apply_rate > 0.3:
            return True

    # 조건 2: 월 폭주 (하루 ~2.7건)
    if max_monthly is not None and max_monthly > 60:
        return True

    # 조건 3: 주 폭주 (하루 ~4.3건)
    if max_weekly is not None and max_weekly > 30:
        return True

    return False


def filter_heavy_users(
    users_df: pd.DataFrame,
    total_jobs: int,
    verbose: bool = True
) -> pd.DataFrame:
    """
    과다지원자(heavy users)를 필터링하여 제거

    Args:
        users_df: 사용자 DataFrame (BOARD_IDX 컬럼 필요)
        total_jobs: 전체 공고 수
        verbose: 로그 출력 여부

    Returns:
        필터링된 DataFrame
    """
    if "BOARD_IDX" not in users_df.columns:
        if verbose:
            print("[WARN] BOARD_IDX 컬럼이 없어 과다지원자 필터링을 건너뜁니다.")
        return users_df

    original_count = len(users_df)

    # 각 사용자의 총 지원 횟수 계산
    def count_applies(board_idx_str):
        if pd.isna(board_idx_str) or not str(board_idx_str).strip():
            return 0
        return len([p for p in str(board_idx_str).split("|") if p.strip()])

    users_df = users_df.copy()
    users_df["_total_applies"] = users_df["BOARD_IDX"].apply(count_applies)

    # max_monthly, max_weekly 컬럼이 있으면 사용, 없으면 None
    max_monthly_col = users_df.get("MAX_MONTHLY_APPLIES")
    max_weekly_col = users_df.get("MAX_WEEKLY_APPLIES")

    # 과다지원자 마스크 생성
    heavy_mask = users_df.apply(
        lambda row: is_heavy_user(
            total_applies=row["_total_applies"],
            total_jobs=total_jobs,
            max_monthly=row.get("MAX_MONTHLY_APPLIES") if max_monthly_col is not None else None,
            max_weekly=row.get("MAX_WEEKLY_APPLIES") if max_weekly_col is not None else None,
        ),
        axis=1
    )

    filtered_df = users_df[~heavy_mask].drop(columns=["_total_applies"])
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    if verbose:
        print(f"[INFO] 과다지원자 필터링: {original_count} → {filtered_count} (제거: {removed_count}명)")

    return filtered_df


def build_training_pairs(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    negative_ratio: int,
    random_seed: int,
) -> pd.DataFrame:
    """Build training pairs with positive labels from user applications and negative sampling.

    Only users and jobs with all required fields are considered. Distances are computed as
    integer km. spec_match is 1 if user's normalized specialty exists in job's normalized
    specialties set, otherwise 0.
    """
    rng = np.random.default_rng(random_seed)

    jobs_filtered = jobs_df.dropna(subset=["ORG_lat", "ORG_lon", "PAY"]).copy()
    jobs_filtered["BOARD_IDX"] = jobs_filtered["BOARD_IDX"].astype(int)

    jobs_filtered["SPECIALTIES_SET"] = jobs_filtered["SPECIALTIES"].apply(parse_job_specialties_to_set)
    jobs_filtered = jobs_filtered.dropna(subset=["SPECIALTIES_SET"]).copy()

    # 경력 요구사항 보조 컬럼: 공고 처리 단계에서 생성된 경우 활용
    if "JOB_MIN_CAREER_YEARS" not in jobs_filtered.columns:
        jobs_filtered["JOB_MIN_CAREER_YEARS"] = np.nan
    if "JOB_MAX_CAREER_YEARS" not in jobs_filtered.columns:
        jobs_filtered["JOB_MAX_CAREER_YEARS"] = np.nan
    if "JOB_IS_CAREER_IRRELEVANT" not in jobs_filtered.columns:
        jobs_filtered["JOB_IS_CAREER_IRRELEVANT"] = 1

    # 공고 초대유형 문자열 (INVITE_TYPE) 기본값
    if "INVITE_TYPE" not in jobs_filtered.columns:
        jobs_filtered["INVITE_TYPE"] = None

    # 공고 조직 타입(R_ORG_TYPE) 기본값
    if "R_ORG_TYPE" not in jobs_filtered.columns:
        jobs_filtered["R_ORG_TYPE"] = None

    # SERVICE_6 공고 학습제외: process(03)에서 SERVICE_ORIG 보존 → 여기서 필터링
    # SERVICE_ORIG 값이 "SERVICE_6"인 공고는 제외
    try:
        if "SERVICE_ORIG" in jobs_filtered.columns:
            jobs_filtered = jobs_filtered[jobs_filtered["SERVICE_ORIG"].astype(str).str.strip() != "SERVICE_6"].copy()
    except Exception:
        pass

    board_to_job = jobs_filtered.set_index("BOARD_IDX")[
        [
            "SPECIALTIES_SET",
            "PAY",
            "ORG_lat",
            "ORG_lon",
            "JOB_MIN_CAREER_YEARS",
            "JOB_MAX_CAREER_YEARS",
            "JOB_IS_CAREER_IRRELEVANT",
            "INVITE_TYPE",
            "R_ORG_TYPE",
        ]
    ]

    # 직장 좌표(geo_OFFICE_ADDR_lat/lon)는 필수가 아님 - 없으면 distance_office = NaN
    required_user_cols = [
        "U_ID",
        "SPECIALTY",
        "geo_ADDR_lat",
        "geo_ADDR_lon",
        # "geo_OFFICE_ADDR_lat",  # 선택사항 - 없으면 distance_office = NaN
        # "geo_OFFICE_ADDR_lon",  # 선택사항 - 없으면 distance_office = NaN
        "CAREER_YEARS",
    ]

    users_filtered = users_df.dropna(subset=required_user_cols).copy()
    users_filtered["SPECIALTY_NORM"] = users_filtered["SPECIALTY"].apply(normalize_specialty_label)
    users_filtered = users_filtered.dropna(subset=["SPECIALTY_NORM"]).copy()

    # 과다지원자(heavy user) 필터링 적용
    total_jobs = len(jobs_filtered)
    users_filtered = filter_heavy_users(users_filtered, total_jobs=total_jobs, verbose=True)

    records: List[dict] = []

    # Parse embedding vectors (JSON -> normalized np.array)
    def _parse_vec(v):
        try:
            if isinstance(v, str) and v:
                arr = np.asarray(json.loads(v), dtype=float)
                norm = np.linalg.norm(arr)
                return arr / norm if norm > 0 else None
        except Exception:
            return None
        return None

    users_filtered = users_filtered.copy()
    jobs_filtered = jobs_filtered.copy()
    # Gemini embedding (3072d) 또는 기존 OpenAI embedding (4096d) 지원
    resume_emb_col = "RESUME_EMB_3072" if "RESUME_EMB_3072" in users_filtered.columns else "RESUME_EMB_4096"
    job_emb_col = "JOB_EMB_3072" if "JOB_EMB_3072" in jobs_filtered.columns else "JOB_EMB_4096"
    users_filtered["_RESUME_VEC"] = users_filtered.get(resume_emb_col, pd.Series([None]*len(users_filtered))).apply(_parse_vec)
    jobs_filtered["_JOB_VEC"] = jobs_filtered.get(job_emb_col, pd.Series([None]*len(jobs_filtered))).apply(_parse_vec)

    for _, u in users_filtered.iterrows():
        doctor_id = str(u["U_ID"]).strip()
        user_spec = str(u["SPECIALTY_NORM"]).strip()
        home_lat = float(u["geo_ADDR_lat"])  # type: ignore[arg-type]
        home_lon = float(u["geo_ADDR_lon"])  # type: ignore[arg-type]

        # 직장 좌표: 없으면 None (distance_office = NaN으로 처리됨)
        office_lat_raw = u.get("geo_OFFICE_ADDR_lat")
        office_lon_raw = u.get("geo_OFFICE_ADDR_lon")
        if pd.notna(office_lat_raw) and pd.notna(office_lon_raw):
            office_lat = float(office_lat_raw)
            office_lon = float(office_lon_raw)
        else:
            office_lat = None
            office_lon = None

        career_years = u["CAREER_YEARS"]

        applied_set = parse_user_applied_board_ids(u.get("BOARD_IDX"))
        applied_set = set([bid for bid in applied_set if bid in board_to_job.index])

        if len(applied_set) == 0:
            continue

        positives = list(applied_set)

        job_index_set = set(board_to_job.index.tolist())
        negative_candidates = list(job_index_set.difference(applied_set))
        if not negative_candidates:
            continue

        num_neg = min(negative_ratio * len(positives), len(negative_candidates))
        if num_neg <= 0:
            continue

        negatives = list(rng.choice(negative_candidates, size=num_neg, replace=False))

        for board_id in positives + negatives:
            job_row = board_to_job.loc[board_id]
            job_specs: Set[str] = job_row["SPECIALTIES_SET"]  # type: ignore[assignment]
            org_lat = float(job_row["ORG_lat"])  # type: ignore[arg-type]
            org_lon = float(job_row["ORG_lon"])  # type: ignore[arg-type]
            pay = job_row["PAY"]
            job_min_career = job_row.get("JOB_MIN_CAREER_YEARS")
            job_max_career = job_row.get("JOB_MAX_CAREER_YEARS")
            job_irrelevant = int(job_row.get("JOB_IS_CAREER_IRRELEVANT", 1))
            job_invite_type = str(job_row.get("INVITE_TYPE") or "").strip()
            job_r_org_type = str(job_row.get("R_ORG_TYPE") or "").strip()

            distance_home, distance_office = compute_distances_for_board(
                home_lat, home_lon, office_lat, office_lon, org_lat, org_lon
            )

            spec_match = 1 if user_spec in job_specs else 0
            # cosine similarity if available
            sim_val = None
            try:
                u_vec = u.get("_RESUME_VEC")
                # locate job row in jobs_filtered by board_id
                j_idx = jobs_filtered.index[jobs_filtered["BOARD_IDX"].astype(int) == int(board_id)]
                j_vec = jobs_filtered.loc[j_idx, "_JOB_VEC"].iloc[0] if len(j_idx) > 0 else None
            except Exception:
                u_vec, j_vec = None, None
            if u_vec is not None and j_vec is not None:
                try:
                    sim_val = float(np.dot(u_vec, j_vec))
                except Exception:
                    sim_val = None
            applied_label = 1 if board_id in applied_set else 0

            # WORK_TYPE: 공고 INVITE_TYPE과 사용자 U_WORK_TYPE 일치 여부
            u_work_type_raw = u.get("U_WORK_TYPE")
            u_work_type = str(u_work_type_raw).strip() if u_work_type_raw is not None and not (isinstance(u_work_type_raw, float) and np.isnan(u_work_type_raw)) else ""
            work_type = 1 if (job_invite_type and u_work_type and (job_invite_type == u_work_type)) else 0

            # ORG_MATCHING: 공고 R_ORG_TYPE과 사용자 U_ORG_TYPE 일치 여부
            u_org_type_raw = u.get("U_ORG_TYPE")
            u_org_type = str(u_org_type_raw).strip() if u_org_type_raw is not None and not (isinstance(u_org_type_raw, float) and np.isnan(u_org_type_raw)) else ""
            org_matching = 1 if (job_r_org_type and u_org_type and (job_r_org_type == u_org_type)) else 0

            # -----------------------------
            # 경력 매칭 특성 계산
            # -----------------------------
            def _to_int_or_none(x):
                try:
                    if pd.isna(x):
                        return None
                except Exception:
                    pass
                try:
                    return int(x)
                except Exception:
                    return None

            min_req = _to_int_or_none(job_min_career)
            max_req = _to_int_or_none(job_max_career)

            if job_irrelevant == 1:
                career_irrelevant = 1
                career_match = 1
                career_gap = 0
            else:
                career_irrelevant = 0
                if min_req is not None and max_req is not None:
                    c = int(career_years)
                    career_match = 1 if (c >= min_req and c <= max_req) else 0
                    if c < min_req:
                        career_gap = abs(c - min_req)  # 경력 미달 (abs 적용)
                    elif c > max_req:
                        career_gap = abs(c - max_req)  # 경력 초과 (abs 적용)
                    else:
                        career_gap = 0
                elif min_req is not None and max_req is None:
                    c = int(career_years)
                    career_match = 1 if c >= min_req else 0
                    career_gap = 0 if c >= min_req else abs(c - min_req)  # 경력 미달 (abs 적용)
                else:
                    # 조건 파싱 불가능 → 조건 없음으로 간주
                    career_irrelevant = 1
                    career_match = 1
                    career_gap = 0

            # distance_office: NaN이면 그대로 유지 (XGBoost가 처리)
            distance_office_val = distance_office if pd.isna(distance_office) else int(distance_office)
            has_office = 0 if pd.isna(distance_office) else 1

            records.append(
                {
                    "doctor_id": doctor_id,
                    "board_id": int(board_id),
                    "spec_match": int(spec_match),
                    "distance_home": int(distance_home),
                    "distance_office": distance_office_val,  # NaN 가능
                    "has_office": has_office,  # 직장좌표 유무 플래그
                    "CAREER_YEARS": career_years,
                    "PAY": pay,
                    "career_match": int(career_match),
                    "career_gap": int(career_gap),
                    "is_career_irrelevant": int(career_irrelevant),
                    "WORK_TYPE": int(work_type),
                    "ORG_MATCHING": int(org_matching),
                    "similarity": sim_val,
                    "applied": int(applied_label),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "index",
                "doctor_id",
                "board_id",
                "spec_match",
                "distance_home",
                "distance_office",
                "has_office",
                "CAREER_YEARS",
                "PAY",
                "career_match",
                "career_gap",
                "is_career_irrelevant",
                "WORK_TYPE",
                "ORG_MATCHING",
                "similarity",
                "applied",
            ]
        )

    result = pd.DataFrame.from_records(records)
    # distance_office는 NaN 허용 (직장좌표 없는 유저)
    result = result.dropna(
        subset=[
            "doctor_id",
            "board_id",
            "spec_match",
            "distance_home",
            # "distance_office",  # NaN 허용
            "CAREER_YEARS",
            "PAY",
        ]
    ).copy()
    result = result.reset_index(drop=True)
    result.insert(0, "index", result.index.astype(int))
    result = result[[
        "index",
        "doctor_id",
        "board_id",
        "spec_match",
        "distance_home",
        "distance_office",
        "has_office",
        "CAREER_YEARS",
        "PAY",
        "career_match",
        "career_gap",
        "is_career_irrelevant",
        "WORK_TYPE",
        "ORG_MATCHING",
        "similarity",
        "applied",
    ]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_csv",
        type=str,
        default="/SPO/Project/RecSys/data/processed/user_features_20250925_134845/user_features_processed.csv",
    )
    parser.add_argument(
        "--job_csv",
        type=str,
        default="/SPO/Project/RecSys/data/processed/job_features_20250925_134845/job_training_view.csv",
    )
    parser.add_argument("--negative_ratio", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/SPO/Project/RecSys/data/processed/training",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    users_df = pd.read_csv(args.user_csv)
    jobs_df = pd.read_csv(args.job_csv)

    training_df = build_training_pairs(
        users_df=users_df,
        jobs_df=jobs_df,
        negative_ratio=args.negative_ratio,
        random_seed=args.random_seed,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.out_dir, f"training_pairs_{ts}.csv")
    # 전체 학습 컬럼을 모두 저장하고 similarity를 추가 컬럼으로 포함 (옵션 제거)
    training_df.to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()

