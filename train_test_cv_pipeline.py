#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Tuple
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent  # RecmdSys/


def run(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    # 고정 날짜 구간
#    TRAIN_DATE_START = "2024-09-01"
    TRAIN_DATE_START = "2025-03-01"
    TRAIN_DATE_END = "2025-08-31"
    TEST_DATE_START = "2025-09-01"
    TEST_DATE_END = "2025-09-30"

    # 출력 루트(작업 디렉토리 = 현재 실행 디렉토리)
    work_dir = Path.cwd()
    data_raw = work_dir / "data" / "raw"
    data_proc = work_dir / "data" / "processed"
    data_train = work_dir / "data" / "training"
    models_dir = work_dir / "models"
    infer_out = work_dir / "infer"

    # 1) Raw 파싱 - train
    env_train = os.environ.copy()
    env_train["TRAIN_DATE_START"] = TRAIN_DATE_START
    env_train["TRAIN_DATE_END"] = TRAIN_DATE_END
    raw_train_dir = data_raw / "train"
    run([sys.executable, str(PROJECT_ROOT / "process" / "01.parse_raw_dataset.py"), "--out_dir", str(raw_train_dir)], cwd=work_dir, env=env_train)

    # 1-2) Raw 파싱 - test
    env_test = os.environ.copy()
    env_test["TRAIN_DATE_START"] = TEST_DATE_START
    env_test["TRAIN_DATE_END"] = TEST_DATE_END
    raw_test_dir = data_raw / "test"
    run([sys.executable, str(PROJECT_ROOT / "process" / "01.parse_raw_dataset.py"), "--out_dir", str(raw_test_dir)], cwd=work_dir, env=env_test)

    # 2) User features 처리 (train/test)
    run([sys.executable, str(PROJECT_ROOT / "process" / "02.process_user_features.py"),
         "--input", str(raw_train_dir / "user_features.csv"), "--out_dir", str(data_proc / "train")], cwd=work_dir)
    run([sys.executable, str(PROJECT_ROOT / "process" / "02.process_user_features.py"),
         "--input", str(raw_test_dir / "user_features.csv"), "--out_dir", str(data_proc / "test")], cwd=work_dir)

    # 3) Job features 처리 (train/test)
    run([sys.executable, str(PROJECT_ROOT / "process" / "03.process_job_features.py"),
         "--input", str(raw_train_dir / "job_features.csv"), "--out_dir", str(data_proc / "train"), "--concurrency", "50", "--log-interval", "500"], cwd=work_dir)
    run([sys.executable, str(PROJECT_ROOT / "process" / "03.process_job_features.py"),
         "--input", str(raw_test_dir / "job_features.csv"), "--out_dir", str(data_proc / "test"), "--concurrency", "50", "--log-interval", "500"], cwd=work_dir)

    # 4) Training pairs 병합 (train/test)
    run([sys.executable, str(PROJECT_ROOT / "process" / "04.merge_to_training_table.py"),
         "--user_csv", str(data_proc / "train" / "user_features_processed.csv"),
         "--job_csv", str(data_proc / "train" / "job_training_view.csv"),
         "--out_dir", str(data_train / "train")], cwd=work_dir)
    run([sys.executable, str(PROJECT_ROOT / "process" / "04.merge_to_training_table.py"),
         "--user_csv", str(data_proc / "test" / "user_features_processed.csv"),
         "--job_csv", str(data_proc / "test" / "job_training_view.csv"),
         "--out_dir", str(data_train / "test")], cwd=work_dir)

    # 5) Optuna 기반 5-fold 튜닝 (XGBoost)
    from pathlib import Path as _Path
    def latest_csv(dir_path: _Path, prefix: str) -> _Path:
        cands = sorted(dir_path.glob(f"{prefix}_*.csv"), reverse=True)
        if not cands:
            raise FileNotFoundError(f"{dir_path} 내에 {prefix}_*.csv 파일이 없습니다.")
        return cands[0]

    train_csv = latest_csv(data_train / "train", "training_pairs")
    models_dir.mkdir(parents=True, exist_ok=True)
    # Optuna 튜닝: 5-fold 평균 AUC 최대화 파라미터 탐색 후 xgb_tuning.json 저장
    run([sys.executable, str(PROJECT_ROOT / "process" / "05.train_models.py"),
         "--input_csv", str(train_csv),
         "--out_dir", str(models_dir),
         "--models", "xgb",
         "--cv_folds", "5",
         "--tune",
         "--tune_models", "xgb",
         "--tune_trials", "30"], cwd=work_dir)

    # 5-1) Fold별 XGBoost SHAP 분석 (실패해도 파이프라인 계속 진행)
    try:
        for k in range(1, 100):
            fold_dir = models_dir / f"fold_{k}"
            if not fold_dir.exists():
                if k == 1:
                    # no fold output
                    pass
                break
            model_path = fold_dir / "xgb_model.json"
            feats_path = fold_dir / "data_info.json"
            out_dir_shap = fold_dir / "shap_xgb"
            if model_path.exists() and feats_path.exists():
                os.makedirs(out_dir_shap, exist_ok=True)
                try:
                    run([
                        sys.executable,
                        str(PROJECT_ROOT / "tools" / "shap_analysis.py"),
                        "--model_type", "xgb",
                        "--model_path", str(model_path),
                        "--input_csv", str(train_csv),
                        "--output_dir", str(out_dir_shap),
                        "--features_json", str(feats_path),
                        "--sample_n", "50000",
                    ], cwd=work_dir)
                except Exception:
                    # ignore SHAP failures per fold
                    pass
    except Exception:
        pass

    # 6) 튠된 best 파라미터로 전체 데이터 재학습 -> 최종 모델로 inference
    # 튜닝 결과 경로 확인 (내용은 학습 스크립트에서 직접 사용)
    tuning_json = models_dir / "xgb_tuning.json"
    if not tuning_json.exists():
        raise FileNotFoundError(f"튜닝 결과 파일을 찾을 수 없습니다: {tuning_json}")

    # 전체 학습: CV 튠 결과를 그대로 사용하여 학습 스크립트로 위임
    full_train_csv = latest_csv(data_train / "train", "training_pairs")
    final_dir = models_dir / "final"
    run([
        sys.executable,
        str(PROJECT_ROOT / "process" / "05.train_models.py"),
        "--input_csv", str(full_train_csv),
        "--out_dir", str(final_dir),
        "--models", "xgb",
        "--xgb_tuning_json", str(tuning_json),
    ], cwd=work_dir)

    # 6-1) 최종 모델 XGBoost SHAP 분석 (실패해도 계속)
    try:
        out_dir_shap_final = final_dir / "shap_xgb"
        os.makedirs(out_dir_shap_final, exist_ok=True)
        run([
            sys.executable,
            str(PROJECT_ROOT / "tools" / "shap_analysis.py"),
            "--model_type", "xgb",
            "--model_path", str(final_dir / "xgb_full_model.json"),
            "--input_csv", str(full_train_csv),
            "--output_dir", str(out_dir_shap_final),
            "--features_json", str(final_dir / "data_info_full.json"),
            "--sample_n", "50000",
        ], cwd=work_dir)
    except Exception:
        pass

    # 7) 최종 모델로 Test inference: thresholds 0.5~0.9 & calibration(=test CSV)
    test_csv = latest_csv(data_train / "test", "training_pairs")
    model_json = final_dir / "xgb_full_model.json"
    features_json = final_dir / "data_info_full.json"

    infer_out.mkdir(parents=True, exist_ok=True)
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_csv = infer_out / f"inferenced_{thr}.output.csv"
        run([sys.executable, str(PROJECT_ROOT / "tools" / "infer_xgb.py"),
             "--model_json", str(model_json),
             "--input_csv", str(test_csv),
             "--output_csv", str(out_csv),
             "--features_json", str(features_json),
             "--threshold", str(thr)], cwd=work_dir)

    # calibration (test CSV로 보정)
    run([sys.executable, str(PROJECT_ROOT / "tools" / "infer_xgb.py"),
         "--model_json", str(model_json),
         "--input_csv", str(test_csv),
         "--output_csv", str(infer_out / "inferenced_calibration.csv"),
         "--features_json", str(features_json),
         "--calibrate_csv", str(test_csv)], cwd=work_dir)

    # 7) 요약 CSV 생성
    run([sys.executable, str(PROJECT_ROOT / "tools" / "aggregate_infer_meta.py"),
         "--meta_dir", str(infer_out),
         "--out_csv", str(infer_out / "metrics_summary.csv")], cwd=work_dir)

    print(str(infer_out))

    # 8) 로지스틱 회귀는 05.train_models.py의 --models logi로 실행 (옵션 추가 없이)
    try:
        full_train_csv = latest_csv(data_train / "train", "training_pairs")
        run([sys.executable, str(PROJECT_ROOT / "process" / "05.train_models.py"),
             "--input_csv", str(full_train_csv),
             "--out_dir", str(models_dir),
             "--models", "logi",
             "--test_size", "0.2",
             "--random_seed", "42"], cwd=work_dir)
    except Exception as _e:
        try:
            with open(models_dir / "final" / "logi_error.txt", "w") as f:
                f.write(str(_e))
        except Exception:
            pass


if __name__ == "__main__":
    main()


