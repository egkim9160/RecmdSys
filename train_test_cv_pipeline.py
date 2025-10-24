#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent  # RecmdSys/


def run(cmd: list[str], *, cwd: Path, env: dict | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    # 고정 날짜 구간
    TRAIN_DATE_START = "2024-09-01"
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

    # 5) CV(5-fold) 강제 학습 (XGBoost)
    from pathlib import Path as _Path
    def latest_csv(dir_path: _Path, prefix: str) -> _Path:
        cands = sorted(dir_path.glob(f"{prefix}_*.csv"), reverse=True)
        if not cands:
            raise FileNotFoundError(f"{dir_path} 내에 {prefix}_*.csv 파일이 없습니다.")
        return cands[0]

    train_csv = latest_csv(data_train / "train", "training_pairs")
    models_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(PROJECT_ROOT / "process" / "05.train_models.py"),
         "--input_csv", str(train_csv),
         "--out_dir", str(models_dir),
         "--models", "xgb",
         "--cv_folds", "5"], cwd=work_dir)

    # 6) Test inference: thresholds 0.5~0.9 & calibration(=test CSV)
    test_csv = latest_csv(data_train / "test", "training_pairs")
    # fold_5 모델 사용
    model_json = models_dir / "fold_5" / "xgb_model.json"
    if not model_json.exists():
        folds = sorted(models_dir.glob("fold_*"))
        if not folds:
            raise FileNotFoundError("CV 결과 모델을 찾지 못했습니다 (models/fold_*/xgb_model.json)")
        model_json = folds[-1] / "xgb_model.json"

    infer_out.mkdir(parents=True, exist_ok=True)
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out_csv = infer_out / f"inferenced_{thr}.output.csv"
        run([sys.executable, str(PROJECT_ROOT / "tools" / "infer_xgb.py"),
             "--model_json", str(model_json),
             "--input_csv", str(test_csv),
             "--output_csv", str(out_csv),
             "--threshold", str(thr)], cwd=work_dir)

    # calibration (test CSV로 보정)
    run([sys.executable, str(PROJECT_ROOT / "tools" / "infer_xgb.py"),
         "--model_json", str(model_json),
         "--input_csv", str(test_csv),
         "--output_csv", str(infer_out / "inferenced_calibration.csv"),
         "--calibrate_csv", str(test_csv)], cwd=work_dir)

    # 7) 요약 CSV 생성
    run([sys.executable, str(PROJECT_ROOT / "tools" / "aggregate_infer_meta.py"),
         "--meta_dir", str(infer_out),
         "--out_csv", str(infer_out / "metrics_summary.csv")], cwd=work_dir)

    print(str(infer_out))


if __name__ == "__main__":
    main()


