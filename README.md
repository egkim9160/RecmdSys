# RecSys - 추천 시스템 파이프라인

의료 인력 채용 공고와 의료진 매칭을 위한 추천 시스템 파이프라인입니다.

## 프로젝트 구조

```
.
├── run_pipeline.py                  # 파이프라인 통합 실행 스크립트
├── process/                         # 파이프라인 단계별 스크립트
│   ├── 01.parse_raw_dataset.py      # 원시 데이터셋 파싱
│   ├── 02.process_user_features.py  # 사용자 특성 처리
│   ├── 03.process_job_features.py   # 공고 특성 처리
│   ├── 04.merge_to_training_table.py # 학습 데이터 병합
│   └── 05.train_models.py           # 모델 학습
├── tools/                           # 독립 실행 도구
│   ├── infer_xgb.py                 # XGBoost 추론
│   └── shap_analysis.py             # SHAP 분석
├── module/                          # 공통 유틸리티 모듈
│   ├── __init__.py
│   ├── db_utils.py                  # MySQL 연결 유틸리티
│   ├── html_utils.py                # HTML 정제 유틸리티
│   ├── llm_utils.py                 # OpenAI LLM 클라이언트 유틸리티
│   ├── geo_utils.py                 # 지오코딩 유틸리티
│   ├── data_utils.py                # 데이터 처리 유틸리티
│   └── naver_geo.py                 # 네이버 지오코딩 API
├── data/                            # 데이터 디렉토리
│   ├── raw/                         # 원시 데이터 (01번 출력)
│   ├── processed/                   # 처리된 데이터 (02, 03번 출력)
│   └── training/                    # 학습 데이터 (04번 출력)
├── models/                          # 학습된 모델 (05번 출력)
├── .env                             # 환경 변수 설정 파일
└── README.md                        # 본 문서
```

## 환경 설정

### 1. 필수 패키지 설치

```bash
pip install pandas numpy mysql-connector-python python-dotenv
pip install google-cloud-bigquery db-dtypes beautifulsoup4
pip install openai requests
pip install xgboost lightgbm scikit-learn
pip install optuna shap  # 선택적
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 정보를 설정하세요:

```bash
# API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json

# Naver API
NAVER_API_CLIENT_ID=your_naver_client_id
NAVER_API_CLIENT_SECRET=your_naver_client_secret

# MySQL Database
DB_HOST=your_db_host
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DB_CHARSET=utf8mb4

# LLM Models
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # 선택적

# Training Parameters (선택적)
TRAIN_DATE_START=2024-09-01
TRAIN_DATE_END=2025-08-31
```

## 파이프라인 실행

### 통합 실행 (권장)

Python 스크립트로 전체 파이프라인을 한번에 실행:

```bash
# 전체 파이프라인 실행
python run_pipeline.py

# 상세 로그 모드
python run_pipeline.py --verbose

# 특정 단계만 실행
python run_pipeline.py --only 3          # 3번만 실행
python run_pipeline.py --only 5 -v       # 5번만 실행 (상세 모드)

# 특정 단계 스킵
python run_pipeline.py --skip 1,2        # 1,2번 스킵하고 3번부터
```

### 개별 단계 실행

각 단계를 개별적으로 실행할 수도 있습니다:

### 1단계: 원시 데이터 파싱

MySQL 및 BigQuery에서 공고 및 사용자 데이터를 조회하여 CSV로 저장합니다.

```bash
python process/01.parse_raw_dataset.py
```

**출력:**
- `data/raw/job_features.csv` - 공고 특성
- `data/raw/user_features.csv` - 사용자 특성

**주요 기능:**
- 공고 정보 조회 (제목, 내용, 급여, 전문과, 위치 등)
- 지원 사용자 목록 조회
- 사용자 이력서, 선호 조건, 경력 정보 조회
- BigQuery에서 전문과 정보 조회
- HTML 정제 및 텍스트 추출

### 2단계: 사용자 특성 처리

사용자 데이터를 가공하여 학습 가능한 형태로 변환합니다.

```bash
python process/02.process_user_features.py \
    --input data/raw/user_features.csv \
    --out_dir data/processed \
    --workers 8 \
    --verbose
```

**주요 옵션:**
- `--input`: 입력 CSV 경로
- `--limit N`: 처리할 행 수 제한 (테스트용)
- `--workers N`: 지오코딩 병렬 워커 수 (기본: 8)
- `--sleep S`: 지오코딩 호출 간 대기 시간 (초)
- `--no-llm`: LLM 주소 정제 비활성화
- `--verbose`: 진행 로그 출력
- `--embed_model`: 임베딩 모델명 (기본: text-embedding-3-large)
- `--embed_batch`: 임베딩 배치 크기 (기본: 64)

**출력:**
- `data/processed/user_features_{timestamp}/user_features_processed.csv`
- `data/processed/user_features_{timestamp}/NOTES.txt`

**주요 기능:**
- 전문과 정규화 (내과 세부분과 → 내과)
- 주소 지오코딩 (Naver API + LLM 정제)
- 경력 년수 숫자화
- 이력서 텍스트 임베딩 (4096차원)

### 3단계: 공고 특성 처리

공고 데이터를 가공하여 학습 가능한 형태로 변환합니다.

```bash
python process/03.process_job_features.py \
    --input data/raw/job_features.csv \
    --out_dir data/processed \
    --concurrency 50 \
    --verbose
```

**주요 옵션:**
- `--input`: 입력 CSV 경로
- `--limit N`: 처리할 행 수 제한
- `--concurrency N`: 지오코딩 동시 실행 한도 (기본: 20)
- `--no-llm`: LLM 주소 정제 비활성화
- `--verbose`: 진행 로그 출력
- `--embed_model`: 임베딩 모델명
- `--no-embed`: 임베딩 비활성화

**출력:**
- `data/processed/job_features_{timestamp}/job_training_view.csv`
- `data/processed/job_features_{timestamp}/NOTES.txt`
- `data/processed/job_features_{timestamp}/geocoding_logs.jsonl`
- `data/processed/job_features_{timestamp}/llm_logs.jsonl`

**주요 기능:**
- 급여 정규화 (Net 월급, Gross 연봉, Day 일급 → 월급 기준)
- 주소 지오코딩 (ADDRESS → LLM 정제 → REGION)
- 경력 요구사항 파싱
- 공고 텍스트 임베딩 (제목 + 내용)

### 4단계: 학습 데이터 병합

사용자-공고 쌍 데이터를 생성합니다 (Positive + Negative Sampling).

```bash
python process/04.merge_to_training_table.py \
    --user_csv data/processed/user_features_processed.csv \
    --job_csv data/processed/job_training_view.csv \
    --out_dir data/training \
    --negative_ratio 3
```

**주요 옵션:**
- `--user_csv`: 처리된 사용자 CSV 경로
- `--job_csv`: 처리된 공고 CSV 경로
- `--negative_ratio`: Negative 샘플링 비율 (기본: 3)
- `--random_seed`: 랜덤 시드 (기본: 42)

**출력:**
- `data/processed/training/training_pairs_{timestamp}.csv`

**생성 특성:**
- `spec_match`: 전문과 일치 여부 (0/1)
- `distance_home`: 집-공고 위치 거리 (km)
- `distance_office`: 사무실-공고 위치 거리 (km)
- `CAREER_YEARS`: 경력 년수
- `PAY`: 급여 (월급 기준)
- `career_match`: 경력 요구사항 일치 여부
- `career_gap`: 경력 차이
- `is_career_irrelevant`: 경력 무관 여부
- `similarity`: 이력서-공고 임베딩 유사도 (코사인)
- `applied`: 지원 여부 (레이블, 0/1)

### 5단계: 모델 학습

XGBoost, LightGBM, Logistic Regression 모델을 학습합니다.

```bash
# 기본 학습 (홀드아웃)
python process/05.train_models.py \
    --csv data/training/training_pairs_{timestamp}.csv \
    --out_dir models \
    --models all \
    --test_size 0.2

# K-Fold Cross Validation
python process/05.train_models.py \
    --csv data/training/training_pairs_{timestamp}.csv \
    --out_dir models \
    --models all \
    --cv_folds 5 \
    --group_by_doctor

# 하이퍼파라미터 튜닝 (Optuna)
python process/05.train_models.py \
    --csv data/training/training_pairs_{timestamp}.csv \
    --out_dir models \
    --tune \
    --tune_models all \
    --tune_trials 50 \
    --cv_folds 3

# 튜닝 결과 적용
python process/05.train_models.py \
    --csv data/training/training_pairs_{timestamp}.csv \
    --out_dir models \
    --models all \
    --xgb_tuning_json models/xgb_tuning.json \
    --lgbm_tuning_json models/lgbm_tuning.json
```

**주요 옵션:**
- `--models`: 학습할 모델 (all, xgb, lgbm, logi)
- `--test_size`: 검증 데이터 비율 (기본: 0.2)
- `--cv_folds N`: K-Fold CV (0=홀드아웃, >0=K-Fold)
- `--group_by_doctor`: doctor_id 기준 그룹 분할
- `--use_gpu`: GPU 가속 활성화
- `--tune`: 하이퍼파라미터 튜닝 모드
- `--tune_trials N`: 튜닝 시도 횟수

**출력:**
- `models/xgb_model.json` - XGBoost 모델
- `models/lgbm_model.txt` - LightGBM 모델
- `models/*_metrics.json` - 평가 지표
- `models/*_feature_importances.csv` - 특성 중요도
- `models/*_roc_curve.csv` - ROC 곡선 데이터
- `models/*_pr_curve.csv` - PR 곡선 데이터

### 추가: 추론 및 분석 도구

학습 완료 후 독립적으로 실행 가능한 도구들:

```bash
# XGBoost 추론
python tools/infer_xgb.py \
    --model models/xgb_model.json \
    --input data/training/training_pairs_{timestamp}.csv

# SHAP 분석 (특성 중요도 해석)
python tools/shap_analysis.py \
    --model_path models/xgb_model.json \
    --data_path data/training/training_pairs_{timestamp}.csv \
    --model_type xgb \
    --output_dir models/shap_analysis
```

## 모듈 설명

### module/db_utils.py
- `get_connection()`: MySQL 연결 생성

### module/html_utils.py
- `clean_html_and_get_urls(html_string)`: HTML → 정제된 텍스트 변환

### module/llm_utils.py
- `get_openai_client()`: OpenAI 클라이언트 생성
- `get_embedding_client()`: 임베딩 클라이언트 생성
- `batch_embed_texts(client, texts, model, batch_size)`: 배치 임베딩
- `clean_address_with_llm(raw_address, client)`: LLM 주소 정제

### module/geo_utils.py
- `try_geocode(address, cache, sleep_sec, lock)`: 네이버 지오코딩

### module/data_utils.py
- `to_str_safe(value)`: 안전한 문자열 변환
- `normalize_career_years(value)`: 경력 텍스트 → 숫자 변환
- `compute_specialty(major, detail)`: 전문과 계산

### module/naver_geo.py
- `geocode_naver(address)`: 네이버 지오코딩 API 호출
- `get_directions5_summary(start, goal)`: 경로 탐색
- `calculate_haversine_distance(coord1, coord2)`: Haversine 거리 계산

## 주요 특징

1. **모듈화된 구조**: 공통 기능을 `module/` 디렉토리로 분리하여 재사용성 향상
2. **LLM 기반 주소 정제**: 복잡한 한국 주소를 LLM으로 정제 후 지오코딩
3. **임베딩 기반 유사도**: 이력서와 공고 텍스트를 임베딩하여 코사인 유사도 계산
4. **다양한 모델 지원**: XGBoost, LightGBM, Logistic Regression
5. **하이퍼파라미터 튜닝**: Optuna 기반 자동 튜닝
6. **K-Fold CV**: Group-based 분할 지원

## 라이선스

내부 프로젝트

## 작성자

Eun Gyo Kim
