#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User 월급 예측 스크립트

사용자(의사) 정보가 담긴 user_features.csv를 읽어서 각 user의 예상 월급을 예측합니다.
DB에서 최근 1년간의 연봉 통계를 조회하여 조건에 맞는 평균 연봉을 기반으로 예측합니다.

입력: user_features.csv (U_ID, SPECIALTY, U_ORG_TYPE, 지역 정보 포함)
출력: user_salary_predictions.csv (U_ID, PREDICTED_PAY 컬럼)

사용 예:
python predict_user_salary.py --input /path/to/user_features.csv --output /path/to/output.csv
"""

import os
import sys
import argparse
import re
from datetime import date, timedelta
from typing import Optional, Dict, Tuple
import pandas as pd
import pymysql
import pymysql.cursors


# DB 연결 정보 (get_salary_statistics 함수 참고)
DB_CONFIG = {
    'host': '211.218.144.132',
    'port': 3306,
    'user': 'medigate',
    'password': 'apelWkd',
    'database': 'social',
    'charset': 'euckr'
}


def extract_region_from_address(address: Optional[str]) -> Optional[str]:
    """
    주소에서 지역명을 추출합니다.
    예: "서울특별시 강남구..." -> "서울", "경상북도 구미시..." -> "경북"

    가능한 지역 목록: ["경북", "전북", "부산", "경기", "서울", "대구", "울산",
                      "충남", "경남", "전남", "충북", "인천", "강원", "광주", "대전", "세종", "제주"]
    """
    if not isinstance(address, str) or not address.strip():
        return None

    addr = address.strip()

    # 지역 매핑 (광역시/도 -> 축약형)
    region_mapping = {
        '서울': '서울',
        '부산': '부산',
        '대구': '대구',
        '인천': '인천',
        '광주': '광주',
        '대전': '대전',
        '울산': '울산',
        '세종': '세종',
        '경기': '경기',
        '강원': '강원',
        '충북': '충북',
        '충남': '충남',
        '전북': '전북',
        '전남': '전남',
        '경북': '경북',
        '경남': '경남',
        '제주': '제주',
        # 전체 이름 매핑
        '서울특별시': '서울',
        '부산광역시': '부산',
        '대구광역시': '대구',
        '인천광역시': '인천',
        '광주광역시': '광주',
        '대전광역시': '대전',
        '울산광역시': '울산',
        '세종특별자치시': '세종',
        '경기도': '경기',
        '강원도': '강원',
        '충청북도': '충북',
        '충청남도': '충남',
        '전라북도': '전북',
        '전라남도': '전남',
        '경상북도': '경북',
        '경상남도': '경남',
        '제주특별자치도': '제주',
    }

    # 매핑 테이블에서 찾기
    for key, value in region_mapping.items():
        if key in addr:
            return value

    return None


def query_salary_from_db(
    specialty: Optional[str] = None,
    org_type: Optional[str] = None,
    region: Optional[str] = None,
    pay_type: str = 'gross'
) -> Optional[float]:
    """
    DB에서 조건에 맞는 평균 연봉을 조회합니다.

    Args:
        specialty: 전문과 (예: "내과", "외과" 등)
        org_type: 병원 타입 (예: "종합병원", "병원", "의원" 등)
        region: 지역 (예: "서울", "경기", "부산" 등)
        pay_type: 'gross'(세전연봉) 또는 'net'(세후월급)

    Returns:
        평균 연봉(만원 단위, gross) 또는 평균 월급(만원 단위, net). 데이터 없으면 None.
    """
    if pay_type.lower() not in ['gross', 'net']:
        return None

    pay_column = "GROSS_PAY" if pay_type.lower() == 'gross' else "NET_PAY"

    # 최근 1년 기간 설정
    start_date_from = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_date_to = date.today().strftime('%Y-%m-%d')

    # WHERE 조건 구성
    where_clauses = []
    params = []

    if specialty:
        where_clauses.append("SPC_NAME = %s")
        params.append(specialty)

    if org_type:
        where_clauses.append("ORG_TYPE_NAME = %s")
        params.append(org_type)

    if region:
        where_clauses.append("HOP_LOC_NAME LIKE %s")
        params.append(f"%{region}%")

    # 기간 조건 추가
    where_clauses.append("START_DATE >= %s")
    params.append(start_date_from)
    where_clauses.append("START_DATE <= %s")
    params.append(start_date_to)

    # 쿼리 구성
    base_query = f"SELECT AVG({pay_column}) as avg_pay, COUNT(*) as count FROM social.SALARY_RECJOB"
    query = f"{base_query} WHERE {' AND '.join(where_clauses)}"

    try:
        with pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                result = cur.fetchone()

                if result and result['count'] > 0 and result['avg_pay'] is not None:
                    # 만원 단위로 반올림
                    return round(result['avg_pay'])
                else:
                    return None
    except Exception as e:
        print(f"DB 쿼리 오류: {e}")
        return None


def predict_user_salary(user_row: pd.Series) -> Optional[float]:
    """
    단일 user의 예상 월급을 예측합니다.

    Args:
        user_row: user_features의 한 행 (SPECIALTY, U_ORG_TYPE, 주소 등 포함)

    Returns:
        예상 월급(만원 단위). 예측 불가능하면 None.
    """
    specialty = user_row.get('SPECIALTY')
    org_type = user_row.get('U_ORG_TYPE')

    # 주소에서 지역 추출 (여러 컬럼 시도)
    region = None
    for addr_col in ['U_HOME_ADDR', 'U_HOME_ADDRESS', 'R_ADDRESS', 'U_OFFICE_ADDR', 'U_OFFICE_ADDRESS']:
        addr = user_row.get(addr_col)
        if addr:
            region = extract_region_from_address(addr)
            if region:
                break

    # 우선순위 1: 전문과 + 병원타입 + 지역
    if specialty and org_type and region:
        salary = query_salary_from_db(specialty=specialty, org_type=org_type, region=region, pay_type='gross')
        if salary:
            # gross 연봉을 월급으로 변환 (간단히 12로 나눔)
            return round(salary / 12)

    # 우선순위 2: 전문과 + 병원타입
    if specialty and org_type:
        salary = query_salary_from_db(specialty=specialty, org_type=org_type, pay_type='gross')
        if salary:
            return round(salary / 12)

    # 우선순위 3: 전문과 + 지역
    if specialty and region:
        salary = query_salary_from_db(specialty=specialty, region=region, pay_type='gross')
        if salary:
            return round(salary / 12)

    # 우선순위 4: 전문과만
    if specialty:
        salary = query_salary_from_db(specialty=specialty, pay_type='gross')
        if salary:
            return round(salary / 12)

    # 우선순위 5: 병원타입 + 지역
    if org_type and region:
        salary = query_salary_from_db(org_type=org_type, region=region, pay_type='gross')
        if salary:
            return round(salary / 12)

    # 우선순위 6: 병원타입만
    if org_type:
        salary = query_salary_from_db(org_type=org_type, pay_type='gross')
        if salary:
            return round(salary / 12)

    # 조건에 맞는 데이터가 없으면 None 반환
    return None


def main():
    parser = argparse.ArgumentParser(description="User 월급 예측 스크립트")
    parser.add_argument("--input", required=True, help="입력 user_features CSV 경로")
    parser.add_argument("--output", default="user_salary_predictions.csv", help="출력 CSV 경로")
    parser.add_argument("--verbose", action="store_true", help="진행 상황 출력")
    args = parser.parse_args()

    # CSV 읽기
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일이 존재하지 않습니다: {args.input}")
        return

    df = pd.read_csv(args.input)

    if args.verbose:
        print(f"입력 파일: {args.input}")
        print(f"총 {len(df)}명의 user 데이터 로드됨")
        print(f"컬럼: {list(df.columns)}")

    # U_ID 컬럼 확인
    if 'U_ID' not in df.columns:
        print("오류: U_ID 컬럼이 없습니다.")
        return

    # 각 user에 대해 월급 예측
    predictions = []
    for idx, row in df.iterrows():
        u_id = row['U_ID']
        predicted_pay = predict_user_salary(row)
        predictions.append({
            'U_ID': u_id,
            'PREDICTED_PAY': predicted_pay
        })

        if args.verbose and (idx + 1) % 100 == 0:
            print(f"처리 진행: {idx + 1}/{len(df)}")

    # 결과 DataFrame 생성
    result_df = pd.DataFrame(predictions)

    # 예측 성공률 계산
    success_count = result_df['PREDICTED_PAY'].notna().sum()
    success_rate = (success_count / len(result_df)) * 100

    # 결과 저장
    result_df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\n완료!")
    print(f"출력 파일: {args.output}")
    print(f"총 {len(result_df)}명 중 {success_count}명 예측 성공 ({success_rate:.1f}%)")

    if success_count > 0:
        print(f"\n예측된 월급 통계:")
        print(f"  평균: {result_df['PREDICTED_PAY'].mean():.0f}만원")
        print(f"  중앙값: {result_df['PREDICTED_PAY'].median():.0f}만원")
        print(f"  최소: {result_df['PREDICTED_PAY'].min():.0f}만원")
        print(f"  최대: {result_df['PREDICTED_PAY'].max():.0f}만원")


if __name__ == "__main__":
    main()
