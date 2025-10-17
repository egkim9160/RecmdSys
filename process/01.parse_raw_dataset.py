import os
import sys
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
import mysql.connector
from mysql.connector.connection import MySQLConnection
from dotenv import load_dotenv
from google.cloud import bigquery
from bs4 import BeautifulSoup
import html
import re
import time
try:
    import db_dtypes  # noqa: F401
except Exception as e:
    raise RuntimeError("db-dtypes 패키지가 필요합니다. requirements 설치를 확인하세요.") from e

def _load_env_from_project_root() -> None:
    """run_pipeline.py가 위치한 프로젝트 루트(RecmdSys)에서 .env를 로드"""
    try:
        project_root = Path(__file__).resolve().parent.parent  # RecmdSys/
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        # 조용히 무시 (환경 변수는 시스템에서 가져오도록)
        pass

_load_env_from_project_root()

DATE_START = os.getenv("TRAIN_DATE_START", "2025-08-01")
DATE_END = os.getenv("TRAIN_DATE_END", "2025-08-31")

# Ensure project root (RecmdSys/) is on sys.path so that `module/*` can be imported
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass


JOBS_QUERY = f"""
WITH RECURSIVE ApplyTypesUnpivoted AS (
    SELECT
        BOARD_IDX,
        SUBSTRING_INDEX(REPLACE(APPLY_TYPE, ' ', ''), ',', 1) AS APPLY_CODE,
        SUBSTRING(REPLACE(APPLY_TYPE, ' ', ''), LENGTH(SUBSTRING_INDEX(REPLACE(APPLY_TYPE, ' ', ''), ',', 1)) + 2) AS REMAINDER
    FROM CBIZ_RECJOB_BACKUP
    WHERE APPROVAL_FLAG = 'Y' AND DISPLAY_FLAG = 'Y' AND DEL_FLAG = 'N' AND APPLY_TYPE IS NOT NULL AND APPLY_TYPE != ''

    UNION ALL

    SELECT
        BOARD_IDX,
        SUBSTRING_INDEX(REMAINDER, ',', 1),
        SUBSTRING(REMAINDER, LENGTH(SUBSTRING_INDEX(REMAINDER, ',', 1)) + 2)
    FROM ApplyTypesUnpivoted
    WHERE REMAINDER != ''
),
BaseJobs AS (
    SELECT
        J.BOARD_IDX, J.TITLE, J.CONTENT, J.PAY_TYPE, J.PAY, J.INCENTIVE_FLAG,
        J.WORK_HOUR, J.S_WORK_FLAG, J.S_WORK_HOUR, J.H_WORK_FLAG, J.H_WORK_HOUR,
        J.NIGHT_WORK_FLAG, J.NIGHT_PAY_FLAG, J.WEEKEND_WORK_FLAG, J.WEEKEND_PAY_FLAG,
        J.MEAL_FLAG, J.HOUSE_FLAG, J.ATTEND_FLAG, J.INSU_FLAG, J.REGULAR_FLAG,
        J.U_ID, J.REG_DATE, J.START_DATE, J.END_DATE, J.EXPIRE_DATE,
        J.CAREER_TYPE, J.INVITE_TYPE, J.RED_WORK, J.REC_REASON, J.APPLY_TYPE
    FROM CBIZ_RECJOB_BACKUP J
    WHERE J.START_DATE >= '{DATE_START}'
      AND J.END_DATE   <= '{DATE_END}'
      AND J.APPROVAL_FLAG = 'Y'
      AND J.DISPLAY_FLAG  = 'Y'
      AND J.DEL_FLAG      = 'N'
      AND EXISTS (
          SELECT 1
          FROM ApplyTypesUnpivoted AP
          WHERE AP.BOARD_IDX = J.BOARD_IDX
            AND AP.APPLY_CODE IN ('MG', 'CF', 'FF')
      )
),
SpecialtiesAgg AS (
    SELECT
        RMAP.BOARD_IDX,
        GROUP_CONCAT(DISTINCT CM_SPC.CODE_NAME SEPARATOR ', ') AS SPECIALTIES
    FROM CBIZ_RECJOB_MAP_BACKUP RMAP
    JOIN BaseJobs BJ ON RMAP.BOARD_IDX = BJ.BOARD_IDX -- 처리 대상을 BaseJobs로 한정
    JOIN CODE_MASTER CM_SPC ON RMAP.MAP_CODE = CM_SPC.CODE AND CM_SPC.KBN = 'SPC'
    WHERE RMAP.MAP_TYPE = 'SPC' AND CM_SPC.CODE_NAME > ''
    GROUP BY RMAP.BOARD_IDX
),
ApplyMethodsAgg AS (
    SELECT
        AP.BOARD_IDX,
        GROUP_CONCAT(DISTINCT CM.CODE_NAME SEPARATOR ', ') AS APPLY_METHODS
    FROM ApplyTypesUnpivoted AP
    JOIN BaseJobs BJ ON AP.BOARD_IDX = BJ.BOARD_IDX -- 처리 대상을 BaseJobs로 한정
    JOIN CODE_MASTER CM ON AP.APPLY_CODE = CM.CODE AND CM.KBN = 'RECRUIT_APPLY_TYPE'
    GROUP BY AP.BOARD_IDX
),
JobWithCM AS (
    SELECT
        BJ.BOARD_IDX, BJ.TITLE, BJ.CONTENT,
        SA.SPECIALTIES, AMA.APPLY_METHODS,
        CASE
            WHEN BJ.PAY_TYPE = 5 THEN 'Net(세후) 월급'
            WHEN BJ.PAY_TYPE = 6 THEN 'Gross(세전) 연봉'
            WHEN BJ.PAY_TYPE = 7 THEN 'Day 일급'
            ELSE '급여 정보 없음'
        END AS PAY_TYPE_NAME,
        CASE WHEN BJ.PAY_TYPE = 7 THEN CAST(BJ.PAY * 10000 AS UNSIGNED) ELSE NULL END AS PAY_DAY,
        CASE WHEN BJ.PAY_TYPE = 5 THEN CAST(BJ.PAY * 10000 AS UNSIGNED)
             WHEN BJ.PAY_TYPE = 6 THEN CAST(ROUND(BJ.PAY / 12, 0) * 10000 AS UNSIGNED)
             ELSE NULL END AS PAY_MONTH,
        CASE WHEN BJ.PAY_TYPE = 5 THEN CAST(BJ.PAY * 12 * 10000 AS UNSIGNED)
             WHEN BJ.PAY_TYPE = 6 THEN CAST(BJ.PAY * 10000 AS UNSIGNED)
             ELSE NULL END AS PAY_YEAR,
        CASE WHEN BJ.INCENTIVE_FLAG = 'Y' THEN '인센티브 있음' ELSE '인센티브 없음' END AS INCENTIVE_STATUS,
        RC.org_name AS ORGANIZATION_NAME, RC.hos_addr AS ADDRESS,
        BJ.START_DATE, BJ.END_DATE, BJ.EXPIRE_DATE,
        CM_ORG_TYPE.CODE_NAME AS ORG_TYPE_NAME,
        CM_ZON.CODE_NAME AS ZON_NAME,
        CM_SGG.CODE_NAME AS CITY_NAME,
        CM_CAREER.CODE_NAME AS CAREER_REQ_DESC,
        CM_INVITE.CODE_NAME AS INVITE_TYPE_DESC,
        CM_REASON.CODE_NAME AS REASON_FOR_RECRUITMENT,
        COALESCE(CM_PAY_NET.CODE_NAME, CM_PAY_GROSS.CODE_NAME, CM_PAY_DAY.CODE_NAME) AS PAY_VIEW
    FROM
        BaseJobs BJ
        LEFT JOIN RECRUIT_COMPANY RC ON BJ.U_ID = RC.u_id
        LEFT JOIN SpecialtiesAgg SA ON BJ.BOARD_IDX = SA.BOARD_IDX
        LEFT JOIN ApplyMethodsAgg AMA ON BJ.BOARD_IDX = AMA.BOARD_IDX
        LEFT JOIN CODE_MASTER CM_ORG_TYPE ON RC.org_type = CM_ORG_TYPE.CODE AND CM_ORG_TYPE.KBN = 'RECRUIT_ORG_TYPE_ALL'
        LEFT JOIN CODE_MASTER CM_ZON ON RC.hop_loc_code = CM_ZON.CODE AND CM_ZON.KBN = 'ZON'
        LEFT JOIN CODE_MASTER CM_SGG ON RC.hop_city_code = CM_SGG.CODE AND CM_SGG.KBN = 'SGG'
        LEFT JOIN CODE_MASTER CM_CAREER ON BJ.CAREER_TYPE = CM_CAREER.CODE AND CM_CAREER.KBN = 'RECRUIT_CAREER_RANGE_JOB'
        LEFT JOIN CODE_MASTER CM_INVITE ON BJ.INVITE_TYPE = CM_INVITE.CODE AND CM_INVITE.KBN = 'RECRUIT_INVITE_TYPE_SEARCH'
        LEFT JOIN CODE_MASTER CM_REASON ON BJ.REC_REASON = CM_REASON.CODE AND CM_REASON.KBN = 'RECRUIT_REC_REASON'
        LEFT JOIN CODE_MASTER CM_PAY_NET   ON BJ.PAY = CM_PAY_NET.CODE   AND BJ.PAY_TYPE = 5 AND CM_PAY_NET.KBN = 'RECRUIT_NET_PAY_TYPE'
        LEFT JOIN CODE_MASTER CM_PAY_GROSS ON BJ.PAY = CM_PAY_GROSS.CODE AND BJ.PAY_TYPE = 6 AND CM_PAY_GROSS.KBN = 'RECRUIT_GROSS_PAY_TYPE'
        LEFT JOIN CODE_MASTER CM_PAY_DAY   ON BJ.PAY = CM_PAY_DAY.CODE   AND BJ.PAY_TYPE = 7 AND CM_PAY_DAY.KBN = 'RECRUIT_DAY_PAY_TYPE'
),
JobDetailsCalculated AS (
    SELECT
        JW.BOARD_IDX, JW.TITLE, JW.CONTENT, JW.SPECIALTIES, JW.APPLY_METHODS,
        JW.PAY_VIEW, JW.PAY_TYPE_NAME, JW.PAY_DAY, JW.PAY_MONTH, JW.PAY_YEAR,
        JW.INCENTIVE_STATUS, JW.ORGANIZATION_NAME, JW.ORG_TYPE_NAME,
        CASE
            WHEN JW.ZON_NAME IS NOT NULL AND JW.CITY_NAME IS NOT NULL THEN CONCAT(JW.ZON_NAME, ' ', JW.CITY_NAME)
            WHEN JW.ZON_NAME IS NOT NULL THEN JW.ZON_NAME
            WHEN JW.CITY_NAME IS NOT NULL THEN JW.CITY_NAME
            ELSE NULL
        END AS REGION,
        JW.ADDRESS, JW.START_DATE, JW.END_DATE, JW.EXPIRE_DATE,
        JW.CAREER_REQ_DESC, JW.INVITE_TYPE_DESC, JW.REASON_FOR_RECRUITMENT
    FROM JobWithCM JW
)
SELECT * FROM JobDetailsCalculated;
"""


APPLIED_USERS_QUERY = f"""
SELECT DISTINCT
  RA.BOARD_IDX,
  RA.U_ID
FROM RECRUIT_APPLY RA
JOIN (
  SELECT BJ.BOARD_IDX
  FROM CBIZ_RECJOB_BACKUP BJ
  WHERE BJ.START_DATE >= '{DATE_START}'
    AND BJ.END_DATE   <= '{DATE_END}'
    AND BJ.APPROVAL_FLAG = 'Y'
    AND BJ.DISPLAY_FLAG  = 'Y'
    AND BJ.DEL_FLAG      = 'N'
    AND (
      FIND_IN_SET('MG', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
      FIND_IN_SET('CF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0 OR
      FIND_IN_SET('FF', REPLACE(BJ.APPLY_TYPE, ' ', '')) > 0
    )
) SJ ON RA.BOARD_IDX = SJ.BOARD_IDX;
"""


PERSONALIZED_INFO_QUERY = """
WITH DefaultResume AS (
    SELECT RESUME_IDX, U_ID
    FROM RESUME
    WHERE default_flag = 'Y' AND U_ID = %s
)
SELECT
    dr.U_ID,
    hope_data.HOPE_INVITE_TYPES,
    hope_data.HOPE_SPECIALTIES,
    hope_data.HOPE_REGION,
    preferences_data.MATCH_INVITE_TYPES,
    preferences_data.MATCH_SPECIALTIES,
    preferences_data.MATCH_LOC_ZONE_NAMES,
    preferences_data.MATCH_LOC_CITY_NAMES,
    preferences_data.MATCH_ORGANIZATION_TYPES
FROM DefaultResume dr
LEFT JOIN (
    SELECT dr_sub.U_ID,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'IVT' AND cm.KBN = 'IVT' THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_INVITE_TYPES,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'SPC' AND cm.KBN = 'SPC' THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_SPECIALTIES,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'LOC' AND cm.KBN IN ('ZON', 'SGG') THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_REGION
    FROM DefaultResume dr_sub
    LEFT JOIN RESUME_MAP rm ON dr_sub.RESUME_IDX = rm.RESUME_IDX
    LEFT JOIN CODE_MASTER cm ON rm.MAP_CODE = cm.CODE
    GROUP BY dr_sub.U_ID
) AS hope_data ON dr.U_ID = hope_data.U_ID
LEFT JOIN (
    SELECT crm.U_ID,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'IVT' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_INVITE_TYPES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'SPC' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_SPECIALTIES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'ZON' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_LOC_ZONE_NAMES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'SGG' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_LOC_CITY_NAMES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'RECRUIT_ORG_TYPE_SEARCH' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_ORGANIZATION_TYPES
    FROM CBIZ_REC_MATCHING crm
    LEFT JOIN CODE_MASTER cm ON
        (cm.KBN = 'IVT' AND FIND_IN_SET(cm.CODE, crm.IVT_CODE) > 0) OR
        (cm.KBN = 'SPC' AND FIND_IN_SET(cm.CODE, crm.SPC_CODE) > 0) OR
        (cm.KBN = 'ZON' AND FIND_IN_SET(cm.CODE, crm.LOC_CODE) > 0) OR
        (cm.KBN = 'SGG' AND FIND_IN_SET(cm.CODE, crm.CITY_CODE) > 0) OR
        (cm.KBN = 'RECRUIT_ORG_TYPE_SEARCH' AND FIND_IN_SET(cm.CODE, crm.ORG_CODE) > 0)
    WHERE crm.U_ID = %s
    GROUP BY crm.U_ID
) AS preferences_data ON dr.U_ID = preferences_data.U_ID;
"""


RESUME_INFO_QUERY = """
WITH DefaultResume AS (
    SELECT RESUME_IDX, U_ID
    FROM RESUME
    WHERE default_flag = 'Y' AND U_ID = %s
),
CareerMonths AS (
    SELECT rc.U_ID,
           SUM(
               PERIOD_DIFF(
                 DATE_FORMAT(STR_TO_DATE(CONCAT(IFNULL(NULLIF(rc.to_date, ''), DATE_FORMAT(CURDATE(), '%Y.%m')), '.01'), '%Y.%m.%d'), '%Y%m'),
                 DATE_FORMAT(STR_TO_DATE(CONCAT(rc.from_date, '.01'), '%Y.%m.%d'), '%Y%m')
               ) + 1
           ) AS total_months
    FROM RESUME_CAREER rc
    INNER JOIN DefaultResume dr ON rc.RESUME_IDX = dr.RESUME_IDX
    WHERE rc.from_date IS NOT NULL AND rc.from_date != ''
    GROUP BY rc.U_ID
)
SELECT
    dr.U_ID,
    CASE
        WHEN cm.total_months IS NULL THEN '경력 없음'
        WHEN cm.total_months < 12 THEN '1년 미만'
        WHEN cm.total_months < 36 THEN '1~2년'
        WHEN cm.total_months < 60 THEN '3~4년'
        WHEN cm.total_months < 84 THEN '5~6년'
        WHEN cm.total_months < 108 THEN '7~8년'
        WHEN cm.total_months < 120 THEN '9~10년'
        ELSE '10년 이상'
    END AS CAREER_YEARS
FROM DefaultResume dr
LEFT JOIN CareerMonths cm ON dr.U_ID = cm.U_ID;
"""


def get_connection() -> MySQLConnection:
    from module.db_utils import get_connection as db_connect
    return db_connect()


def fetch_dataframe(conn: MySQLConnection, query: str, params: tuple = None) -> pd.DataFrame:
    return pd.read_sql(query, conn, params=params)


def fetch_user_side_features(conn: MySQLConnection, user_ids: List[int], chunk_size: int = 500) -> pd.DataFrame:
    """
    사용자별 feature를 개별 조회 대신 벌크 SQL(CTE + IN 절)로 조회하여 왕복 횟수를 줄입니다.
    - DefaultResume가 없는 사용자는 결과에 나타나지 않을 수 있으므로, 최종적으로 전체 사용자 목록과 left merge하여 모두 포함합니다.
    - 너무 긴 IN 리스트를 피하기 위해 청크로 나눠 실행합니다.
    """
    if not user_ids:
        return pd.DataFrame(columns=[
            "U_ID",
            "HOPE_INVITE_TYPES",
            "HOPE_SPECIALTIES",
            "HOPE_REGION",
            "MATCH_INVITE_TYPES",
            "MATCH_SPECIALTIES",
            "MATCH_LOC_ZONE_NAMES",
            "MATCH_LOC_CITY_NAMES",
            "MATCH_ORGANIZATION_TYPES",
            "CAREER_YEARS",
            "R_ADDRESS",
            "U_HOME_ADDR",
            "U_OFFICE_ADDR",
        ])

    results: List[pd.DataFrame] = []
    for start in range(0, len(user_ids), chunk_size):
        chunk = user_ids[start:start + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        bulk_query = f"""
WITH DefaultResume AS (
    SELECT RESUME_IDX, U_ID
    FROM RESUME
    WHERE default_flag = 'Y' AND U_ID IN ({placeholders})
),
CareerMonths AS (
    SELECT dr.U_ID,
           SUM(
               PERIOD_DIFF(
                 DATE_FORMAT(STR_TO_DATE(CONCAT(IFNULL(NULLIF(rc.to_date, ''), DATE_FORMAT(CURDATE(), '%Y.%m')), '.01'), '%Y.%m.%d'), '%Y%m'),
                 DATE_FORMAT(STR_TO_DATE(CONCAT(rc.from_date, '.01'), '%Y.%m.%d'), '%Y%m')
               ) + 1
           ) AS total_months
    FROM RESUME_CAREER rc
    INNER JOIN DefaultResume dr ON rc.RESUME_IDX = dr.RESUME_IDX
    WHERE rc.from_date IS NOT NULL AND rc.from_date != ''
    GROUP BY dr.U_ID
),
HopeData AS (
    SELECT dr_sub.U_ID,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'IVT' AND cm.KBN = 'IVT' THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_INVITE_TYPES,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'SPC' AND cm.KBN = 'SPC' THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_SPECIALTIES,
           GROUP_CONCAT(DISTINCT CASE WHEN rm.MAP_TYPE = 'LOC' AND cm.KBN IN ('ZON', 'SGG') THEN cm.CODE_NAME END SEPARATOR ', ') AS HOPE_REGION
    FROM DefaultResume dr_sub
    LEFT JOIN RESUME_MAP rm ON dr_sub.RESUME_IDX = rm.RESUME_IDX
    LEFT JOIN CODE_MASTER cm ON rm.MAP_CODE = cm.CODE
    GROUP BY dr_sub.U_ID
),
PreferencesData AS (
    SELECT crm.U_ID,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'IVT' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_INVITE_TYPES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'SPC' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_SPECIALTIES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'ZON' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_LOC_ZONE_NAMES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'SGG' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_LOC_CITY_NAMES,
           GROUP_CONCAT(DISTINCT CASE WHEN cm.KBN = 'RECRUIT_ORG_TYPE_SEARCH' THEN cm.CODE_NAME END ORDER BY cm.CODE_NAME SEPARATOR ', ') AS MATCH_ORGANIZATION_TYPES
    FROM CBIZ_REC_MATCHING crm
    LEFT JOIN CODE_MASTER cm ON
        (cm.KBN = 'IVT' AND FIND_IN_SET(cm.CODE, crm.IVT_CODE) > 0) OR
        (cm.KBN = 'SPC' AND FIND_IN_SET(cm.CODE, crm.SPC_CODE) > 0) OR
        (cm.KBN = 'ZON' AND FIND_IN_SET(cm.CODE, crm.LOC_CODE) > 0) OR
        (cm.KBN = 'SGG' AND FIND_IN_SET(cm.CODE, crm.CITY_CODE) > 0) OR
        (cm.KBN = 'RECRUIT_ORG_TYPE_SEARCH' AND FIND_IN_SET(cm.CODE, crm.ORG_CODE) > 0)
    WHERE crm.U_ID IN ({placeholders})
    GROUP BY crm.U_ID
),
ResumeAddr AS (
    SELECT dr_sub.U_ID,
           r.address AS R_ADDRESS
    FROM DefaultResume dr_sub
    LEFT JOIN RESUME r ON r.RESUME_IDX = dr_sub.RESUME_IDX
),
UserDetail AS (
    SELECT ud.U_ID,
           ud.U_HOME_ADDR,
           ud.U_OFFICE_ADDR
    FROM USER_DETAIL ud
    WHERE ud.U_ID IN ({placeholders})
)
SELECT
    dr.U_ID,
    hd.HOPE_INVITE_TYPES,
    hd.HOPE_SPECIALTIES,
    hd.HOPE_REGION,
    pd.MATCH_INVITE_TYPES,
    pd.MATCH_SPECIALTIES,
    pd.MATCH_LOC_ZONE_NAMES,
    pd.MATCH_LOC_CITY_NAMES,
    pd.MATCH_ORGANIZATION_TYPES,
    ra.R_ADDRESS,
    ud.U_HOME_ADDR,
    ud.U_OFFICE_ADDR,
    CASE
        WHEN cm.total_months IS NULL THEN '경력 없음'
        WHEN cm.total_months < 12 THEN '1년 미만'
        WHEN cm.total_months < 36 THEN '1~2년'
        WHEN cm.total_months < 60 THEN '3~4년'
        WHEN cm.total_months < 84 THEN '5~6년'
        WHEN cm.total_months < 108 THEN '7~8년'
        WHEN cm.total_months < 120 THEN '9~10년'
        ELSE '10년 이상'
    END AS CAREER_YEARS
FROM DefaultResume dr
LEFT JOIN HopeData hd ON dr.U_ID = hd.U_ID
LEFT JOIN PreferencesData pd ON dr.U_ID = pd.U_ID
LEFT JOIN CareerMonths cm ON dr.U_ID = cm.U_ID
LEFT JOIN ResumeAddr ra ON dr.U_ID = ra.U_ID
LEFT JOIN UserDetail ud ON dr.U_ID = ud.U_ID;
"""

        params = tuple(chunk) + tuple(chunk) + tuple(chunk)
        try:
            df_chunk = pd.read_sql(bulk_query, conn, params=params)
        except Exception as e:
            print(f"[warn] 사용자 청크 {len(chunk)}건 벌크 조회 실패: {e}")
            df_chunk = pd.DataFrame(columns=[
                "U_ID",
                "HOPE_INVITE_TYPES",
                "HOPE_SPECIALTIES",
                "HOPE_REGION",
                "MATCH_INVITE_TYPES",
                "MATCH_SPECIALTIES",
                "MATCH_LOC_ZONE_NAMES",
                "MATCH_LOC_CITY_NAMES",
                "MATCH_ORGANIZATION_TYPES",
                "CAREER_YEARS",
                "R_ADDRESS",
                "U_HOME_ADDR",
                "U_OFFICE_ADDR",
            ])
        results.append(df_chunk)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["U_ID"])
    # 모든 입력 사용자 유지 (DefaultResume 미보유자 포함)
    all_users_df = pd.DataFrame({"U_ID": user_ids})
    full_df = all_users_df.merge(combined.drop_duplicates(subset=["U_ID"]), on="U_ID", how="left")
    return full_df


def clean_html_and_get_urls(html_string: str) -> Tuple[str, List[str]]:
    """HTML에서 텍스트를 추출하고 태그를 제거해 문장 단위로 정리합니다. (이미지 URL 미사용)"""
    from module.html_utils import clean_html_and_get_urls as clean_html
    return clean_html(html_string)


def detect_resume_content_columns(conn: MySQLConnection) -> List[str]:
    """
    information_schema를 조회해 RESUME 테이블에서 컨텐츠성 텍스트 컬럼을 자동 탐지합니다.
    현재 정책: TITLE, CONTENT 두 컬럼만 사용합니다(있을 때만).
    """
    sql = """
SELECT COLUMN_NAME
FROM information_schema.columns
WHERE table_schema = DATABASE()
  AND table_name = 'RESUME'
"""
    try:
        df = pd.read_sql(sql, conn)
        cols = set(df['COLUMN_NAME'].astype(str).str.strip().tolist()) if not df.empty else set()
    except Exception:
        cols = set()
    # TITLE, CONTENT만 채택 (대소문자 무시)
    selected = []
    if any(c.lower() == 'title' for c in cols):
        selected.append('TITLE')
    if any(c.lower() == 'content' for c in cols):
        selected.append('CONTENT')
    return selected


def fetch_resume_raw_html_for_users(
    conn: MySQLConnection,
    user_ids: List[int],
    content_columns: List[str],
    chunk_size: int = 500,
) -> pd.DataFrame:
    """
    기본 이력서(DefaultResume)가 있는 사용자에 대해 RESUME 테이블의 컨텐츠성 컬럼들을 벌크로 조회합니다.
    반환: U_ID + 각 컨텐츠 컬럼
    """
    if not user_ids or not content_columns:
        return pd.DataFrame(columns=["U_ID"] + content_columns)

    results: List[pd.DataFrame] = []
    for start in range(0, len(user_ids), chunk_size):
        chunk = user_ids[start:start + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        select_cols_sql = ",\n           ".join([f"r.`{c}` AS `{c}`" for c in content_columns])
        bulk_query = f"""
WITH DefaultResume AS (
    SELECT RESUME_IDX, U_ID
    FROM RESUME
    WHERE default_flag = 'Y' AND U_ID IN ({placeholders})
)
SELECT
    dr.U_ID,
           {select_cols_sql}
FROM DefaultResume dr
LEFT JOIN RESUME r ON r.RESUME_IDX = dr.RESUME_IDX;
"""
        try:
            df_chunk = pd.read_sql(bulk_query, conn, params=tuple(chunk))
        except Exception as e:
            print(f"[warn] 이력서 컨텐츠 벌크 조회 실패: {e}")
            df_chunk = pd.DataFrame(columns=["U_ID"] + content_columns)
        results.append(df_chunk)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["U_ID"] + content_columns)
    return combined


def build_clean_resume_fields(raw_resume_df: pd.DataFrame, preferred_order: List[str]) -> pd.DataFrame:
    """
    RESUME 테이블의 TITLE, CONTENT 두 컬럼만 사용해 텍스트를 생성합니다.
    반환: U_ID, RESUME_TEXT (TITLE + 공백행 + CONTENT 순서)
    """
    if raw_resume_df.empty:
        return pd.DataFrame(columns=["U_ID", "RESUME_TEXT"])

    # TITLE/CONTENT 컬럼만 사용 (없으면 빈 문자열)
    title_col = 'TITLE' if 'TITLE' in raw_resume_df.columns else None
    content_col = 'CONTENT' if 'CONTENT' in raw_resume_df.columns else None

    texts: List[str] = []
    for _, row in raw_resume_df.iterrows():
        title_html = (row.get(title_col) if title_col else "") or ""
        content_html = (row.get(content_col) if content_col else "") or ""
        title_text = clean_html_and_get_urls(str(title_html))[0] if title_html else ""
        content_text = clean_html_and_get_urls(str(content_html))[0] if content_html else ""
        combined = "\n\n".join([t for t in [title_text.strip(), content_text.strip()] if t])
        texts.append(combined)

    out_df = pd.DataFrame({
        "U_ID": raw_resume_df["U_ID"],
        "RESUME_TEXT": texts,
    })
    return out_df


def fetch_specialties_from_bigquery(user_ids: List[int], chunk_size: int = 1000) -> pd.DataFrame:
    """
    BigQuery에서 MG_USERS 테이블을 조회해 사용자별 대표전문과/세부전문과를 벌크로 가져옵니다.
    - GOOGLE_APPLICATION_CREDENTIALS 환경변수가 설정되어 있어야 합니다.
    - 반환 컬럼: U_ID, MAJOR_SPECIALTY, DETAIL_SPECIALTY
    """
    if not user_ids:
        return pd.DataFrame(columns=["U_ID", "MAJOR_SPECIALTY", "DETAIL_SPECIALTY"])

    client = bigquery.Client()
    # 1) 가능한 경우 단일 쿼리로 전체 사용자 조회 (최소 왕복)
    query = """
SELECT U_ID, MAJOR_SPECIALTY, DETAIL_SPECIALTY
FROM `medigate-mate.MEDIGATE.MG_USERS`
WHERE U_ID IN UNNEST(@user_ids)
"""
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("user_ids", "STRING", [str(uid) for uid in user_ids])
        ])
        job = client.query(query, job_config=job_config)
        df_full = job.result().to_dataframe()
        return df_full.drop_duplicates(subset=["U_ID"]) if not df_full.empty else df_full
    except Exception as e:
        print(f"[warn] BigQuery 단일쿼리 실패: {e}. 큰 청크로 폴백합니다...")

    # 2) 단일쿼리 실패 시에만 큰 청크로 최소 횟수 폴백
    large_chunk = max(chunk_size, 20000)
    results: List[pd.DataFrame] = []
    for start in range(0, len(user_ids), large_chunk):
        chunk = user_ids[start:start + large_chunk]
        try:
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ArrayQueryParameter("user_ids", "STRING", [str(uid) for uid in chunk])
            ])
            job = client.query(query, job_config=job_config)
            df = job.result().to_dataframe()
        except Exception as e:
            print(f"[warn] BigQuery 전문과 조회 실패 (chunk {start}~): {e}")
            df = pd.DataFrame(columns=["U_ID", "MAJOR_SPECIALTY", "DETAIL_SPECIALTY"])
        results.append(df)

    combined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["U_ID", "MAJOR_SPECIALTY", "DETAIL_SPECIALTY"])
    return combined.drop_duplicates(subset=["U_ID"]) if not combined.empty else combined


def main():
    print("환경 변수 로드 및 DB 연결 시도...")
    t_start = time.time()
    try:
        conn = get_connection()
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        sys.exit(1)

    print("1) 기간 및 온라인 지원 조건의 공고 feature 조회...")
    t0 = time.time()
    jobs_df = fetch_dataframe(conn, JOBS_QUERY)
    before_cnt = len(jobs_df)
    jobs_df = jobs_df.drop_duplicates(subset=["BOARD_IDX"])  # BOARD_IDX 기준 유니크 보장
    print(f"공고 feature {before_cnt}건 → unique board {len(jobs_df)}건")
    print(f"[time] 1) 공고 feature 조회/중복제거: {time.time() - t0:.2f}s")

    # 1-0) SPECIALTIES 필수 검증: 한 건이라도 누락 시 즉시 에러 발생
    if "SPECIALTIES" not in jobs_df.columns:
        raise RuntimeError("SPECIALTIES 컬럼이 조회 결과에 존재하지 않습니다. SQL을 확인하세요.")
    missing_specialties = jobs_df[jobs_df["SPECIALTIES"].isna() | (jobs_df["SPECIALTIES"].astype(str).str.strip() == "")]
    if not missing_specialties.empty:
        sample_ids = ", ".join(map(str, missing_specialties["BOARD_IDX"].head(10).tolist()))
        raise RuntimeError(f"SPECIALTIES 누락 {len(missing_specialties)}건 발견 (예: {sample_ids}). 매핑 테이블/코드마스터를 점검하세요.")

    # 1-a) 공고 CONTENT 정제 텍스트/이미지 URL 생성
    try:
        if os.getenv("SKIP_CONTENT_CLEAN", "0") == "1":
            print("1-a) 공고 CONTENT 정제 건너뜀 (SKIP_CONTENT_CLEAN=1)")
        else:
            print("1-a) 공고 CONTENT 정제 텍스트/이미지 URL 생성...")
            t1a = time.time()
            if "CONTENT" in jobs_df.columns:
                safe_series = jobs_df["CONTENT"].fillna("").astype(str)
                job_texts: List[str] = []
                job_img_urls: List[str] = []
                for html_str in safe_series.tolist():
                    text, urls = clean_html_and_get_urls(html_str)
                    # CONTENT는 반드시 한 줄로: 줄바꿈 제거 후 공백 정규화
                    one_line_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
                    job_texts.append(one_line_text)
                    # 이미지 URL은 더 이상 사용하지 않음
                # CONTENT를 정제 텍스트로 덮어쓰기 (원본은 저장하지 않음)
                jobs_df["CONTENT"] = job_texts
                jobs_df["JOB_IMAGE_URLS"] = job_img_urls
            else:
                print("[warn] jobs_df에 CONTENT 컬럼이 없어 정제를 건너뜁니다.")
            print(f"[time] 1-a) CONTENT 정제: {time.time() - t1a:.2f}s")
    except Exception as e:
        print(f"[warn] 공고 CONTENT 정제 실패: {e}")

    # 1-b) 공고 컬럼 재정렬 및 CONTENT_RAW 제거
    try:
        print("1-b) 공고 컬럼 재정렬 및 불필요 컬럼 제거...")
        if "CONTENT_RAW" in jobs_df.columns:
            jobs_df = jobs_df.drop(columns=["CONTENT_RAW"])
        tail_cols = [c for c in ["TITLE", "CONTENT"] if c in jobs_df.columns]
        other_cols = [c for c in jobs_df.columns if c not in tail_cols]
        jobs_df = jobs_df[other_cols + tail_cols]
    except Exception as e:
        print(f"[warn] 공고 컬럼 재정렬 실패: {e}")

    # 1-c) 급여 숫자 컬럼을 정수형(Int64, 결측 허용)으로 캐스팅하여 .0 제거
    try:
        t1c = time.time()
        for pay_col in ["PAY_DAY", "PAY_MONTH", "PAY_YEAR"]:
            if pay_col in jobs_df.columns:
                jobs_df[pay_col] = pd.to_numeric(jobs_df[pay_col], errors='coerce').astype('Int64')
        print(f"[time] 1-c) 급여 정수 캐스팅: {time.time() - t1c:.2f}s")
    except Exception as e:
        print(f"[warn] 급여 정수 캐스팅 실패: {e}")

    print("2) 해당 공고에 실제 지원한 사용자 조회...")
    t2 = time.time()
    applied_df = fetch_dataframe(conn, APPLIED_USERS_QUERY)
    print(f"지원 기록 {len(applied_df)}건")
    print(f"[time] 2) 지원 사용자 조회: {time.time() - t2:.2f}s")

    if applied_df.empty or jobs_df.empty:
        print("데이터가 부족하여 종료합니다.")
        conn.close()
        return

    print("3) 사용자 ID 수집 및 BigQuery 전문과/세부전문과 조회...")
    user_ids = applied_df["U_ID"].dropna().unique().tolist()
    print(f"   unique user 수: {len(user_ids)}명")
    # 3-1) BigQuery 조회 먼저 수행
    try:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print("[warn] GOOGLE_APPLICATION_CREDENTIALS 미설정 — BigQuery 인증이 필요합니다. .env에 경로를 추가하세요.")
        t3_1 = time.time()
        bq_spc_df = fetch_specialties_from_bigquery(user_ids)
        print(f"   BigQuery specialty rows: {len(bq_spc_df)}")
        print(f"[time] 3-1) BigQuery 전문과 조회: {time.time() - t3_1:.2f}s")
    except Exception as e:
        print(f"[warn] BigQuery 전문과 조회 실패: {e}")
        bq_spc_df = pd.DataFrame({"U_ID": user_ids, "MAJOR_SPECIALTY": [None]*len(user_ids), "DETAIL_SPECIALTY": [None]*len(user_ids)})

    # 3-2) 사용자 측 feature 조회 (MySQL)
    print("3-2) 사용자 측 feature 조회 (기본 이력서/선호/경력 요약)...")
    t3_2 = time.time()
    user_feat_df = fetch_user_side_features(conn, user_ids)
    print(f"사용자 feature {len(user_feat_df)}명")
    print(f"[time] 3-2) 사용자 feature 조회(MySQL): {time.time() - t3_2:.2f}s")

    # 3-2-추가) 이력서 컨텐츠 컬럼 자동 탐지 및 HTML 정제 텍스트 생성
    try:
        print("3-2-a) RESUME 컨텐츠성 컬럼 자동 탐지...")
        resume_cols = detect_resume_content_columns(conn)
        print(f"   탐지된 컬럼: {', '.join(resume_cols) if resume_cols else '(없음)'}")
        if resume_cols:
            print("3-2-b) 사용자 기본 이력서 컨텐츠 벌크 조회...")
            t3_2a = time.time()
            raw_resume_df = fetch_resume_raw_html_for_users(conn, user_ids, resume_cols)
            print(f"   raw resume rows: {len(raw_resume_df)}")
            print("3-2-c) HTML 정제 및 텍스트/이미지 URL 생성...")
            clean_resume_df = build_clean_resume_fields(raw_resume_df, resume_cols)
            user_feat_df = user_feat_df.merge(clean_resume_df, on="U_ID", how="left")
            print(f"[time] 3-2-a~c) 이력서 컨텐츠 조회+정제: {time.time() - t3_2a:.2f}s")
        else:
            print("[warn] RESUME 테이블에서 컨텐츠성 컬럼을 찾지 못했습니다. RESUME_TEXT 생성을 건너뜁니다.")
    except Exception as e:
        print(f"[warn] 이력서 컨텐츠 정제 파이프라인 실패: {e}")

    # 사용자별 지원 공고 리스트
    print("3-3) 사용자별 지원 공고 리스트 생성...")
    t3_3 = time.time()
    applied_by_user = (
        applied_df.groupby("U_ID")["BOARD_IDX"]
        .apply(lambda s: "|".join(map(str, sorted(set(s)))))
        .reset_index(name="BOARD_IDX")
    )

    # 병합: MySQL features → BigQuery specialties → BOARD_IDX
    user_feat_df = user_feat_df.merge(bq_spc_df, on="U_ID", how="left").merge(applied_by_user, on="U_ID", how="left")
    print(f"[time] 3-3) 지원 리스트 생성+머지: {time.time() - t3_3:.2f}s")

    start_cols = ["U_ID", "MAJOR_SPECIALTY", "DETAIL_SPECIALTY", "R_ADDRESS", "U_HOME_ADDR", "U_OFFICE_ADDR"]
    end_cols = ["BOARD_IDX"]
    existing_cols = [c for c in user_feat_df.columns if c not in start_cols + end_cols]
    user_feat_df = user_feat_df[[c for c in start_cols if c in user_feat_df.columns] + existing_cols + [c for c in end_cols if c in user_feat_df.columns]]

    conn.close()

    print("4) CSV 저장(디버깅 용이성 우선)...")
    t4 = time.time()
    os.makedirs("data/raw", exist_ok=True)
    jobs_df.to_csv("data/raw/job_features.csv", index=False)
    user_feat_df.to_csv("data/raw/user_features.csv", index=False)
    print(f"[time] 4) CSV 저장: {time.time() - t4:.2f}s")

    print("완료: data/raw/ 하위에 job_features.csv, user_features.csv 저장")
    print(f"[time] 총 소요 시간: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()


