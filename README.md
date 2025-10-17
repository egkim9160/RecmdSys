# RecSys - 추천 시스템 파이프라인

의료 인력 채용 공고와 의료진 매칭을 위한 추천 시스템 파이프라인입니다.

## 프로젝트 구조

```
.
├── run_pipeline.py                  # 파이프라인 통합 실행 스크립트
├── process/                         # 파이프라인 단계별 스크립트
│   ├── 01.parse_raw_dataset.py      # 원시 데이터셋 파싱 (DB/BigQuery → CSV)
│   ├── 02.process_user_features.py  # 사용자 특성 처리 (지오코딩, 임베딩)
│   ├── 03.process_job_features.py   # 공고 특성 처리 (급여 정규화, 지오코딩, 임베딩)
│   ├── 04.merge_to_training_table.py # 학습 데이터 병합 (Positive/Negative 샘플링)
│   └── 05.train_models.py           # 모델 학습/검증/튜닝
├── tools/                           # 독립 실행 도구
│   ├── infer_xgb.py                 # XGBoost Booster 추론/임계값 보정
│   └── shap_analysis.py             # XGBoost/LightGBM SHAP 분석 및 요약
├── module/                          # 공통 유틸리티 모듈
│   ├── __init__.py
│   ├── db_utils.py                  # MySQL 연결 유틸리티
│   ├── html_utils.py                # HTML 정제 유틸리티
│   ├── llm_utils.py                 # OpenAI LLM/임베딩 유틸리티
│   ├── geo_utils.py                 # 지오코딩 보조 유틸리티(캐시/락)
│   ├── data_utils.py                # 데이터 처리 유틸리티
│   └── naver_geo.py                 # 네이버 지도 API 래퍼(지오코딩/길찾기)
├── data/                            # 데이터 디렉터리
│   ├── raw/                         # 원시 데이터 (01 출력)
│   ├── processed/                   # 처리된 데이터 (02,03,04 출력)
│   └── training/                    # 학습 데이터 (04 출력)
├── models/                          # 학습된 모델/리포트/튜닝 결과 (05 출력)
├── requirements.txt                 # 의존성 목록
├── .env                             # 환경 변수 설정 파일
└── README.md                        # 본 문서
```

## 설치 및 요구사항

### 1) 패키지 설치

```bash
pip install -r /SPO/Project/RecSys/upload/RecmdSys/requirements.txt
```

- 선택 설치: `optuna`, `statsmodels`, `shap`, `matplotlib`은 선택 항목입니다. 필요 시 유지, 불필요하면 제거 가능합니다.
- LightGBM GPU 사용 시 OpenCL/CUDA 환경이 필요할 수 있습니다(미설치 시 자동 CPU 폴백).

### 2) 환경 변수 설정(.env)

프로젝트 루트(`RecmdSys/`)에 `.env` 파일을 생성하고 다음 정보를 설정하세요:

```bash
# API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1  # 선택적, 임베딩 별도 엔드포인트 사용 가능
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

# Training 기간(예시)
TRAIN_DATE_START=2024-09-01
TRAIN_DATE_END=2025-08-31
```

## 파이프라인 실행

### 통합 실행(권장)

`run_pipeline.py`는 1~5단계를 순차 실행하며, 5단계 학습 모드도 통합 제공합니다.

```bash
# 전체 파이프라인 실행(기본 holdout 학습)
python run_pipeline.py

# 상세 로그 모드
python run_pipeline.py --verbose

# 특정 단계만 실행
python run_pipeline.py --only 3           # 3번만 실행
python run_pipeline.py --only 5 -v        # 5번만 실행(상세)

# 특정 단계 스킵
python run_pipeline.py --skip 1,2         # 1,2 스킵하고 이후 단계 진행

# 5단계 학습 모드 제어
python run_pipeline.py --only 5 --train-mode holdout                # 기본 홀드아웃
python run_pipeline.py --only 5 --train-mode cv --cv-folds 5        # K-Fold CV
python run_pipeline.py --only 5 --train-mode tune --tune-trials 30  # Optuna 튜닝만
python run_pipeline.py --only 5 --train-mode tune+cv --cv-folds 5   # 튜닝 후 CV

# 그룹 분할, GPU 등 옵션(5단계에서 사용)
python run_pipeline.py --only 5 --group-by-doctor --use-gpu --gpu-id 0
```

- 작업 디렉터리: `run_pipeline.py`는 현재 작업 디렉터리(CWD) 기준으로 `data/`, `models/` 등을 생성/사용합니다.
- 5단계는 최신 `data/training/training_pairs_*.csv`를 자동 선택해 학습합니다.

### 개별 단계 실행

각 단계를 개별적으로 실행할 수도 있습니다.

#### 1단계: 원시 데이터 파싱(예: MySQL/BigQuery → CSV)

```bash
python process/01.parse_raw_dataset.py
```

- 출력:
  - `data/raw/job_features.csv`
  - `data/raw/user_features.csv`

#### 2단계: 사용자 특성 처리

```bash
python process/02.process_user_features.py \
    --input /SPO/Project/RecSys/data/raw/user_features.csv \
    --out_dir /SPO/Project/RecSys/data/processed/user_features_$(date +%Y%m%d_%H%M%S) \
    --workers 8 \
    --verbose
```

- 주요 옵션:
  - `--input`: 입력 CSV 경로
  - `--limit N`: 처리할 행 수 제한(테스트용)
  - `--workers N`: 지오코딩 병렬 워커 수(기본: 8)
  - `--sleep S`: 지오코딩 호출 간 대기 시간(초)
  - `--no-llm`: LLM 주소 정제 비활성화
  - `--resume_policy {exclude|fallback|zero}`: 이력서 공백 처리 정책
  - `--embed_model`, `--embed_batch`: 임베딩 설정
- 출력(예시):
  - `data/processed/user_features_{timestamp}/user_features_processed.csv`
  - `data/processed/user_features_{timestamp}/NOTES.txt`

#### 3단계: 공고 특성 처리

```bash
python process/03.process_job_features.py \
    --input /SPO/Project/RecSys/data/raw/job_features.csv \
    --out_dir /SPO/Project/RecSys/data/processed/job_features_{your_ts} \
    --concurrency 50 \
    --log-interval 200 \
    --verbose
```

- 주요 옵션:
  - `--input`: 입력 CSV 경로
  - `--limit N`: 처리할 행 수 제한
  - `--concurrency N`: 지오코딩 동시 실행 한도(기본: 20)
  - `--no-llm`: LLM 주소 정제 비활성화
  - `--embed_model`, `--embed_batch`, `--no-embed`: 임베딩 설정
  - `--log-interval`: 진행 로그 간격
- 출력(예시):
  - `data/processed/job_features_{your_ts}/job_training_view.csv`
  - `data/processed/job_features_{your_ts}/NOTES.txt`
  - `data/processed/job_features_{your_ts}/geocoding_logs.jsonl`
  - `data/processed/job_features_{your_ts}/llm_logs.jsonl`

> 참고: 스크립트 기본값은 고정 타임스탬프 디렉터리를 사용합니다. 재현/관리 목적이 아니라면 `--out_dir`를 명시해 사용하는 것을 권장합니다.

#### 4단계: 학습 데이터 병합

```bash
python process/04.merge_to_training_table.py \
    --user_csv /SPO/Project/RecSys/data/processed/user_features_{ts}/user_features_processed.csv \
    --job_csv  /SPO/Project/RecSys/data/processed/job_features_{ts}/job_training_view.csv \
    --out_dir  /SPO/Project/RecSys/data/training \
    --negative_ratio 3
```

- 주요 옵션:
  - `--user_csv`, `--job_csv`: 처리된 사용자/공고 CSV 경로
  - `--negative_ratio`: Negative 샘플링 비율(기본: 3)
  - `--random_seed`: 랜덤 시드(기본: 42)
- 출력:
  - `data/training/training_pairs_{timestamp}.csv`

- 생성 특성(발췌):
  - `spec_match`, `distance_home`, `distance_office`, `CAREER_YEARS`, `PAY`
  - `career_match`, `career_gap`, `is_career_irrelevant`, `similarity`, `applied`

#### 5단계: 모델 학습/검증/튜닝

```bash
# 기본 학습(홀드아웃)
python process/05.train_models.py \
    --input_csv /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --out_dir /SPO/Project/RecSys/models \
    --models all \
    --test_size 0.2

# K-Fold Cross Validation (group split 지원)
python process/05.train_models.py \
    --input_csv /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --out_dir /SPO/Project/RecSys/models \
    --models all \
    --cv_folds 5 \
    --group_by_doctor

# 하이퍼파라미터 튜닝(Optuna)
python process/05.train_models.py \
    --input_csv /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --out_dir /SPO/Project/RecSys/models \
    --tune \
    --tune_models all \
    --tune_trials 50 \
    --cv_folds 0

# 튜닝 결과 적용 후 재학습(CV)
python process/05.train_models.py \
    --input_csv /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --out_dir /SPO/Project/RecSys/models \
    --models all \
    --cv_folds 5 \
    --xgb_tuning_json /SPO/Project/RecSys/models/xgb_tuning.json \
    --lgbm_tuning_json /SPO/Project/RecSys/models/lgbm_tuning.json
```

- 주요 옵션:
  - `--models`: 학습할 모델(`all`, `xgb`, `lgbm`, `logi`)
  - `--test_size`: 검증 데이터 비율(홀드아웃)
  - `--cv_folds`: 0=홀드아웃, >0=K-Fold
  - `--group_by_doctor`: `doctor_id` 기준 그룹 분할
  - `--use_gpu`, `--gpu_id`: GPU 가속(가능 시)
  - `--tune`, `--tune_models`, `--tune_trials`: Optuna 튜닝
- 출력(발췌):
  - `models/xgb_model.json`, `models/lgbm_model.txt`
  - `models/*_metrics.json`, `*_feature_importances.csv`, `*_roc_curve.csv`, `*_pr_curve.csv`
  - CV 시 `models/fold_*/` 하위에 폴드별 산출물 저장

### 추가 도구: 추론/해석

#### XGBoost Booster 추론 및 임계값 보정

```bash
python tools/infer_xgb.py \
    --model_json /SPO/Project/RecSys/models/xgb_model.json \
    --input_csv  /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --output_csv /SPO/Project/RecSys/models/inferred.csv \
    --features_json /SPO/Project/RecSys/models/data_info.json           # 선택

# 레이블 포함 CSV로 F1 최대 임계값 보정
python tools/infer_xgb.py \
    --model_json /SPO/Project/RecSys/models/xgb_model.json \
    --input_csv  /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --output_csv /SPO/Project/RecSys/models/inferred.csv \
    --calibrate_csv /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv
```

- `data_info.json`에 `features` 배열이 있으면 자동 사용(명시적 `--features_json`로도 지정 가능).

#### SHAP 분석(XGBoost/LightGBM)

```bash
python tools/shap_analysis.py \
    --model_type xgb \
    --model_path /SPO/Project/RecSys/models/xgb_model.json \
    --input_csv  /SPO/Project/RecSys/data/training/training_pairs_{timestamp}.csv \
    --output_dir /SPO/Project/RecSys/models/shap \
    --features_json /SPO/Project/RecSys/models/data_info.json \
    --no_plots
```

- 산출물: `*_shap_values.csv`, `*_shap_summary.csv`, (선택) `*_shap_bar.png`, `*_shap_beeswarm.png`
- 시각화는 `matplotlib`/`shap` 설치 시 생성됩니다(`--no_plots`로 비활성화 가능).

## 모듈 개요

- `module/db_utils.py`: `get_connection()` - MySQL 연결 생성
- `module/html_utils.py`: `clean_html_and_get_urls(html_string)` - HTML → 정제 텍스트
- `module/llm_utils.py`: OpenAI 클라이언트/임베딩/주소 정제 유틸
- `module/geo_utils.py`: `try_geocode(address, cache, sleep_sec, lock)` - 네이버 지오코딩 with 캐시
- `module/data_utils.py`: 문자열/경력/전문과 처리 유틸
- `module/naver_geo.py`: `geocode_naver`, `get_directions5_summary`, `calculate_haversine_distance`

## 주요 특징

1. 모듈화된 구조로 재사용성 향상
2. LLM 기반 주소 정제 후 네이버 지오코딩 재시도
3. 이력서/공고 임베딩 기반 유사도 특성 지원
4. XGBoost/LightGBM/Logistic Regression 모델 및 Optuna 튜닝
5. Group K-Fold 및 SHAP 기반 해석 지원

## 라이선스

내부 프로젝트

## 작성자

Eun Gyo Kim
