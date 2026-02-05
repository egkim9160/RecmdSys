import os
import sys
from typing import List
from pathlib import Path

import pandas as pd


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
        else:
            load_dotenv()
    except Exception:
        pass


def get_conn():
    # 재사용: 기존 db_utils의 SQLAlchemy 커넥션 우선
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from module.db_utils import get_sqlalchemy_connection, get_connection
    try:
        return get_sqlalchemy_connection()
    except Exception:
        return get_connection()


def table_exists(conn, table_name: str) -> bool:
    sql = """
SELECT COUNT(*) AS cnt
FROM information_schema.tables
WHERE table_schema = DATABASE() AND table_name = %s
"""
    try:
        df = pd.read_sql(sql, conn, params=(table_name,))
        return bool(df.iloc[0, 0] > 0)
    except Exception:
        return False


def list_columns(conn, table_name: str) -> List[str]:
    sql = """
SELECT COLUMN_NAME
FROM information_schema.columns
WHERE table_schema = DATABASE() AND table_name = %s
ORDER BY ORDINAL_POSITION
"""
    try:
        df = pd.read_sql(sql, conn, params=(table_name,))
        return df["COLUMN_NAME"].astype(str).str.strip().tolist() if not df.empty else []
    except Exception:
        return []


def list_columns_detail(conn, table_name: str) -> pd.DataFrame:
    sql = """
SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE
FROM information_schema.columns
WHERE table_schema = DATABASE() AND table_name = %s
ORDER BY ORDINAL_POSITION
"""
    try:
        return pd.read_sql(sql, conn, params=(table_name,))
    except Exception:
        return pd.DataFrame(columns=["COLUMN_NAME", "DATA_TYPE", "COLUMN_TYPE", "IS_NULLABLE"])


def run_checks(date_start: str, date_end: str) -> None:
    print(f"[debug] date range: {date_start} ~ {date_end}")
    conn = get_conn()

    required = {
        "RECRUIT_APPLY": ["U_ID", "BOARD_IDX"],
        "CBIZ_RECJOB_BACKUP": ["BOARD_IDX", "START_DATE", "END_DATE", "APPROVAL_FLAG", "DISPLAY_FLAG", "DEL_FLAG", "APPLY_TYPE"],
        "RESUME": ["RESUME_IDX", "U_ID", "default_flag"],
        "RESUME_CAREER": ["RESUME_IDX", "from_date", "to_date"],
        "RESUME_MAP": ["RESUME_IDX", "MAP_TYPE", "MAP_CODE"],
        "CODE_MASTER": ["KBN", "CODE", "CODE_NAME"],
        "CBIZ_REC_MATCHING": ["U_ID", "IVT_CODE", "SPC_CODE", "LOC_CODE", "CITY_CODE", "ORG_CODE"],
        "USERS": ["U_ID"],
        "USER_DETAIL": ["U_ID", "U_HOME_ADDR", "U_OFFICE_ADDR", "U_WORK_TYPE_1", "U_HOSPITAL_GROUP_CODE"],
    }

    print("[debug] 1) 테이블/컬럼 점검")
    for tbl, cols in required.items():
        exists = table_exists(conn, tbl)
        print(f"- table {tbl}: {'OK' if exists else 'MISSING'}")
        if exists:
            actual_cols = list_columns(conn, tbl)
            actual_cols_lc = {c.lower() for c in actual_cols}
            missing = [c for c in cols if c.lower() not in actual_cols_lc]
            if missing:
                print(f"  missing: {', '.join(missing)}")
            # 상세 컬럼 정보 일부 출력(앞 8개)
            detail = list_columns_detail(conn, tbl)
            if not detail.empty:
                preview = ", ".join([f"{r['COLUMN_NAME']}:{r['DATA_TYPE']}" for _, r in detail.head(8).iterrows()])
                print(f"  columns: {preview}{' ...' if len(detail) > 8 else ''}")

    print("[debug] 2) 보드 필터 건수")
    boards_sql = f"""
SELECT COUNT(DISTINCT BJ.BOARD_IDX) AS cnt
FROM CBIZ_RECJOB_BACKUP BJ
WHERE BJ.START_DATE >= '{date_start}'
  AND BJ.END_DATE   <= '{date_end}'
  AND BJ.APPROVAL_FLAG = 'Y'
  AND BJ.DISPLAY_FLAG  = 'Y'
  AND BJ.DEL_FLAG      = 'N'
  AND (
    FIND_IN_SET('MG', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
    FIND_IN_SET('CF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
    FIND_IN_SET('FF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0
  )
"""
    try:
        boards_cnt = int(pd.read_sql(boards_sql, conn).iloc[0, 0])
        print(f"- boards: {boards_cnt}")
    except Exception as e:
        print(f"- boards_sql 실패: {e}")

    print("[debug] 3) AppliedUsers 수")
    applied_users_sql = f"""
SELECT COUNT(DISTINCT RA.U_ID) AS cnt
FROM RECRUIT_APPLY RA
JOIN (
    SELECT BJ.BOARD_IDX
    FROM CBIZ_RECJOB_BACKUP BJ
    WHERE BJ.START_DATE >= '{date_start}'
      AND BJ.END_DATE   <= '{date_end}'
      AND BJ.APPROVAL_FLAG = 'Y'
      AND BJ.DISPLAY_FLAG  = 'Y'
      AND BJ.DEL_FLAG      = 'N'
      AND (
        FIND_IN_SET('MG', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
        FIND_IN_SET('CF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
        FIND_IN_SET('FF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0
      )
) SJ ON RA.BOARD_IDX = SJ.BOARD_IDX
"""
    try:
        applied_users_cnt = int(pd.read_sql(applied_users_sql, conn).iloc[0, 0])
        print(f"- applied users: {applied_users_cnt}")
    except Exception as e:
        print(f"- applied_users_sql 실패: {e}")

    print("[debug] 4) USERS/USER_DETAIL 겹침 및 누락 샘플")
    users_overlap_sql = f"""
SELECT
  COUNT(DISTINCT AU.U_ID) AS applied_users,
  SUM(CASE WHEN U.U_ID IS NOT NULL THEN 1 ELSE 0 END) AS users_overlap
FROM (
  SELECT DISTINCT RA.U_ID
  FROM RECRUIT_APPLY RA
  JOIN (
      SELECT BJ.BOARD_IDX
      FROM CBIZ_RECJOB_BACKUP BJ
      WHERE BJ.START_DATE >= '{date_start}'
        AND BJ.END_DATE   <= '{date_end}'
        AND BJ.APPROVAL_FLAG = 'Y'
        AND BJ.DISPLAY_FLAG  = 'Y'
        AND BJ.DEL_FLAG      = 'N'
        AND (
          FIND_IN_SET('MG', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
          FIND_IN_SET('CF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
          FIND_IN_SET('FF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0
        )
  ) SJ ON RA.BOARD_IDX = SJ.BOARD_IDX
) AU
LEFT JOIN USERS U ON TRIM(U.U_ID) = TRIM(AU.U_ID)
"""
    missing_users_sql = f"""
SELECT AU.U_ID
FROM (
  SELECT DISTINCT RA.U_ID
  FROM RECRUIT_APPLY RA
  JOIN (
      SELECT BJ.BOARD_IDX
      FROM CBIZ_RECJOB_BACKUP BJ
      WHERE BJ.START_DATE >= '{date_start}'
        AND BJ.END_DATE   <= '{date_end}'
        AND BJ.APPROVAL_FLAG = 'Y'
        AND BJ.DISPLAY_FLAG  = 'Y'
        AND BJ.DEL_FLAG      = 'N'
        AND (
          FIND_IN_SET('MG', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
          FIND_IN_SET('CF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
          FIND_IN_SET('FF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0
        )
  ) SJ ON RA.BOARD_IDX = SJ.BOARD_IDX
) AU
LEFT JOIN USERS U ON TRIM(U.U_ID) = TRIM(AU.U_ID)
WHERE U.U_ID IS NULL
LIMIT 10
"""
    try:
        df_overlap = pd.read_sql(users_overlap_sql, conn)
        applied_users = int(df_overlap.iloc[0, 0])
        overlap = int(df_overlap.iloc[0, 1])
        print(f"- USERS overlap: {overlap}/{applied_users}")
        try:
            df_missing = pd.read_sql(missing_users_sql, conn)
            if not df_missing.empty and 'U_ID' in df_missing.columns:
                print("- missing U_ID sample in USERS:", ", ".join(map(str, df_missing["U_ID"].tolist())))
        except Exception as e:
            print(f"- USERS 누락 샘플 조회 실패: {e}")
    except Exception as e:
        print(f"- USERS 겹침 계산 실패: {e}")

    user_detail_overlap_sql = users_overlap_sql.replace("LEFT JOIN USERS U", "LEFT JOIN USER_DETAIL U")
    missing_user_detail_sql = missing_users_sql.replace("LEFT JOIN USERS U", "LEFT JOIN USER_DETAIL U")
    try:
        df_ud = pd.read_sql(user_detail_overlap_sql, conn)
        applied_users = int(df_ud.iloc[0, 0])
        overlap = int(df_ud.iloc[0, 1])
        print(f"- USER_DETAIL overlap: {overlap}/{applied_users}")
        try:
            df_missing_ud = pd.read_sql(missing_user_detail_sql, conn)
            if not df_missing_ud.empty and 'U_ID' in df_missing_ud.columns:
                print("- missing U_ID sample in USER_DETAIL:", ", ".join(map(str, df_missing_ud["U_ID"].tolist())))
        except Exception as e:
            print(f"- USER_DETAIL 누락 샘플 조회 실패: {e}")
    except Exception as e:
        print(f"- USER_DETAIL 겹침 계산 실패: {e}")

    # USER_DETAIL의 U_HOSPITAL_GROUP_CODE 존재율 체크
    try:
        cnt_all = int(pd.read_sql("SELECT COUNT(DISTINCT U_ID) AS c FROM USER_DETAIL", conn).iloc[0, 0])
        cnt_has = int(pd.read_sql("SELECT COUNT(DISTINCT U_ID) AS c FROM USER_DETAIL WHERE U_HOSPITAL_GROUP_CODE IS NOT NULL AND U_HOSPITAL_GROUP_CODE <> ''", conn).iloc[0, 0])
        print(f"[debug] USER_DETAIL U_HOSPITAL_GROUP_CODE 존재: {cnt_has}/{cnt_all}")
    except Exception as e:
        print(f"[debug] USER_DETAIL U_HOSPITAL_GROUP_CODE 점검 실패: {e}")

    print("[debug] 완료")


if __name__ == "__main__":
    _load_env()
    date_start = os.getenv("TRAIN_DATE_START", "2024-09-01")
    date_end = os.getenv("TRAIN_DATE_END", "2025-08-31")
    run_checks(date_start, date_end)


