#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecSys 파이프라인 통합 실행 스크립트

전체 파이프라인을 순차적으로 실행:
1. Raw Dataset 파싱 (DB → CSV)
2. User Features 처리 (임베딩, 지오코딩 등)
3. Job Features 처리 (급여 계산, 지오코딩 등)
4. Training Table 병합
5. 모델 학습

사용법:
    python run_pipeline.py
    python run_pipeline.py --skip 1,2  # 1,2번 스킵하고 3번부터 실행
    python run_pipeline.py --only 3    # 3번만 실행
    python run_pipeline.py --verbose   # 상세 로그 출력
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class Colors:
    """터미널 색상 정의"""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class PipelineRunner:
    """파이프라인 실행 관리 클래스"""

    def __init__(
        self,
        script_dir: Path,
        work_dir: Path,
        *,
        verbose: bool = False,
        train_mode: str = "holdout",  # holdout | cv | tune | tune+cv
        cv_folds: int = 5,
        tune_trials: int = 30,
        group_by_doctor: bool = False,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ):
        self.script_dir = script_dir  # 스크립트 파일들이 있는 디렉토리
        self.work_dir = work_dir      # 작업 디렉토리 (데이터/모델 저장)
        self.verbose = verbose

        # 학습 관련 설정
        self.train_mode = train_mode
        self.cv_folds = int(cv_folds or 0)
        self.tune_trials = int(tune_trials or 30)
        self.group_by_doctor = bool(group_by_doctor)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = int(gpu_id or 0)

        # 경로 설정 (작업 디렉토리 기준)
        self.data_raw_dir = work_dir / "data" / "raw"
        self.data_processed_dir = work_dir / "data" / "processed"
        self.data_training_dir = work_dir / "data" / "training"
        self.models_dir = work_dir / "models"

        # 디렉토리 생성
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 파이프라인 단계 정의 (process/ 디렉토리 하위)
        self.steps = [
            {
                "id": 1,
                "name": "Raw Dataset 파싱",
                "script": "process/01.parse_raw_dataset.py",
                "args": [],
            },
            {
                "id": 2,
                "name": "User Features 처리",
                "script": "process/02.process_user_features.py",
                "args": [
                    "--input", str(self.data_raw_dir / "user_features.csv"),
                    "--out_dir", str(self.data_processed_dir),
                ],
            },
            {
                "id": 3,
                "name": "Job Features 처리",
                "script": "process/03.process_job_features.py",
                "args": [
                    "--input", str(self.data_raw_dir / "job_features.csv"),
                    "--out_dir", str(self.data_processed_dir),
                    "--log-interval", "500",
                    "--concurrency", "50",
                ],
            },
            {
                "id": 4,
                "name": "Training Table 병합",
                "script": "process/04.merge_to_training_table.py",
                "args": [
                    "--user_csv", str(self.data_processed_dir / "user_features_processed.csv"),
                    "--job_csv", str(self.data_processed_dir / "job_training_view.csv"),
                    "--out_dir", str(self.data_training_dir),
                ],
            },
            {
                "id": 5,
                "name": "모델 학습",
                "script": "process/05.train_models.py",
                "args": [],  # CSV 경로는 런타임에 결정
            },
        ]

    def log_info(self, msg: str):
        """INFO 로그 출력"""
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

    def log_error(self, msg: str):
        """ERROR 로그 출력"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

    def log_step(self, msg: str):
        """STEP 로그 출력"""
        print(f"{Colors.YELLOW}[STEP]{Colors.NC} {msg}")

    def log_debug(self, msg: str):
        """DEBUG 로그 출력 (verbose 모드일 때만)"""
        if self.verbose:
            print(f"{Colors.CYAN}[DEBUG]{Colors.NC} {msg}")

    def run_step(self, step: dict) -> bool:
        """단일 파이프라인 단계 실행

        Args:
            step: 단계 정의 딕셔너리

        Returns:
            성공 여부 (True/False)
        """
        step_id = step["id"]
        step_name = step["name"]
        script = step["script"]
        args = step["args"].copy()

        # verbose 플래그 추가 (1~3번 단계만 지원)
        if self.verbose and "--verbose" not in args and step_id in [1, 2, 3]:
            args.append("--verbose")

        # 5번 단계는 최신 training CSV 찾기 및 학습 모드/옵션 처리
        if step_id == 5:
            training_dir = self.data_training_dir
            training_csvs = sorted(training_dir.glob("training_pairs_*.csv"), reverse=True)
            if not training_csvs:
                self.log_error(f"Training CSV를 찾을 수 없습니다: {training_dir}")
                return False
            training_csv = training_csvs[0]
            self.log_debug(f"Training CSV 선택: {training_csv.name}")

            # 학습 실행 모드에 따른 분기 처리
            script_path = str(self.script_dir / script)
            base_common = [
                "--input_csv", str(training_csv),
            ]
            if self.use_gpu:
                base_common += ["--use_gpu", "--gpu_id", str(self.gpu_id)]

            def _run(cmd_args: list[str]) -> bool:
                cmd = ["python", script_path] + cmd_args
                self.log_debug(f"실행 명령어: {' '.join(cmd)}")
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=str(self.work_dir),
                    capture_output=not self.verbose,
                    text=True,
                    check=False,
                )
                elapsed = time.time() - start_time
                if result.returncode == 0:
                    self.log_info(f"{script} 완료 (소요: {elapsed:.1f}초)")
                    if not self.verbose and result.stdout:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 5:
                            print("...")
                        for line in lines[-5:]:
                            print(line)
                    return True
                else:
                    self.log_error(f"{script} 실패 (반환 코드: {result.returncode})")
                    if result.stderr:
                        print(f"{Colors.RED}=== STDERR ==={Colors.NC}")
                        print(result.stderr)
                    if result.stdout:
                        print(f"{Colors.RED}=== STDOUT ==={Colors.NC}")
                        print(result.stdout)
                    return False

            def _run_shap(model_type: str, model_path: Path, output_dir: Path) -> bool:
                shap_script = str(self.script_dir / "tools" / "shap_analysis.py")
                cmd = [
                    "python", shap_script,
                    "--model_type", model_type,
                    "--model_path", str(model_path),
                    "--input_csv", str(training_csv),
                    "--output_dir", str(output_dir),
#                    "--no_plots",
                ]
                self.log_debug(f"SHAP 실행 명령어: {' '.join(cmd)}")
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=str(self.work_dir),
                    capture_output=not self.verbose,
                    text=True,
                    check=False,
                )
                elapsed = time.time() - start_time
                if result.returncode == 0:
                    self.log_info(f"SHAP({model_type}) 계산 완료 (소요: {elapsed:.1f}초, 출력: {output_dir})")
                    if not self.verbose and result.stdout:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 5:
                            print("...")
                        for line in lines[-5:]:
                            print(line)
                    return True
                else:
                    self.log_error(f"SHAP({model_type}) 계산 실패 (반환 코드: {result.returncode})")
                    if result.stderr:
                        print(f"{Colors.RED}=== STDERR ==={Colors.NC}")
                        print(result.stderr)
                    if result.stdout:
                        print(f"{Colors.RED}=== STDOUT ==={Colors.NC}")
                        print(result.stdout)
                    return False

            # 모드: holdout (기존과 동일)
            if self.train_mode == "holdout":
                cmd_args = base_common + [
                    "--out_dir", str(self.models_dir),
                ] + args
                success = _run(cmd_args)
                if not success:
                    return False
                # 학습 성공 시 SHAP 계산 (xgb/lgbm)
                shap_out = self.models_dir / "shap"
                shap_out.mkdir(parents=True, exist_ok=True)
                xgb_model = self.models_dir / "xgb_model.json"
                lgbm_model = self.models_dir / "lgbm_model.txt"
                if xgb_model.exists():
                    _ = _run_shap("xgb", xgb_model, shap_out)
                if lgbm_model.exists():
                    _ = _run_shap("lgbm", lgbm_model, shap_out)
                return True

            # 모드: cv (K-fold)
            if self.train_mode == "cv":
                cmd_args = base_common + [
                    "--out_dir", str(self.models_dir),
                    "--cv_folds", str(max(2, self.cv_folds or 5)),
                ]
                if self.group_by_doctor:
                    cmd_args.append("--group_by_doctor")
                cmd_args += args
                success = _run(cmd_args)
                if not success:
                    return False
                # 각 fold의 모델에 대해 SHAP 계산
                for fold_dir in sorted(self.models_dir.glob("fold_*")):
                    shap_out = fold_dir / "shap"
                    shap_out.mkdir(parents=True, exist_ok=True)
                    xgb_model = fold_dir / "xgb_model.json"
                    lgbm_model = fold_dir / "lgbm_model.txt"
                    if xgb_model.exists():
                        _ = _run_shap("xgb", xgb_model, shap_out)
                    if lgbm_model.exists():
                        _ = _run_shap("lgbm", lgbm_model, shap_out)
                return True

            # 모드: tune (Bayesian optuna 튜닝만 수행) - 홀드아웃(test_size=0.2)
            if self.train_mode == "tune":
                tuning_dir = self.models_dir / "tuning"
                tuning_dir.mkdir(parents=True, exist_ok=True)
                cmd_args = base_common + [
                    "--out_dir", str(tuning_dir),
                    "--tune",
                    "--tune_models", "all",
                    "--tune_trials", str(max(1, self.tune_trials or 30)),
                    "--test_size", "0.2",
                ]
                # group_by_doctor는 CV 전용 옵션이므로 튜닝(홀드아웃)에서는 전달하지 않음
                cmd_args += args
                success = _run(cmd_args)
                return success

            # 모드: tune+cv (run_train.sh 유사 플로우)
            if self.train_mode == "tune+cv":
                tuning_dir = self.models_dir / "tuning"
                tuning_dir.mkdir(parents=True, exist_ok=True)
                # 1) 튜닝
                tune_args = base_common + [
                    "--out_dir", str(tuning_dir),
                    "--tune",
                    "--tune_models", "all",
                    "--tune_trials", str(max(1, self.tune_trials or 30)),
                    "--test_size", "0.2",
                ]
                # group_by_doctor는 CV 전용 옵션이므로 튜닝(홀드아웃)에서는 전달하지 않음
                if not _run(tune_args):
                    return False

                # 2) XGBoost (튜닝 적용 + CV)
                xgb_args = base_common + [
                    "--out_dir", str(self.models_dir),
                    "--models", "xgb",
                    "--cv_folds", str(max(2, self.cv_folds or 5)),
                    "--xgb_tuning_json", str((self.models_dir / "tuning" / "xgb_tuning.json")),
                ]
                if self.group_by_doctor:
                    xgb_args.append("--group_by_doctor")
                if not _run(xgb_args):
                    return False

                # 3) LightGBM (튜닝 적용 + CV)
                lgbm_args = base_common + [
                    "--out_dir", str(self.models_dir),
                    "--models", "lgbm",
                    "--cv_folds", str(max(2, self.cv_folds or 5)),
                    "--lgbm_tuning_json", str((self.models_dir / "tuning" / "lgbm_tuning.json")),
                ]
                if self.group_by_doctor:
                    lgbm_args.append("--group_by_doctor")
                if not _run(lgbm_args):
                    return False

                # 4) Logistic (CV)
                logi_args = base_common + [
                    "--out_dir", str(self.models_dir),
                    "--models", "logi",
                    "--cv_folds", str(max(2, self.cv_folds or 5)),
                ]
                if self.group_by_doctor:
                    logi_args.append("--group_by_doctor")
                success = _run(logi_args)
                if not success:
                    return False
                # 튜닝 후 CV 결과 폴드들에 대해 SHAP 계산
                for fold_dir in sorted(self.models_dir.glob("fold_*")):
                    shap_out = fold_dir / "shap"
                    shap_out.mkdir(parents=True, exist_ok=True)
                    xgb_model = fold_dir / "xgb_model.json"
                    lgbm_model = fold_dir / "lgbm_model.txt"
                    if xgb_model.exists():
                        _ = _run_shap("xgb", xgb_model, shap_out)
                    if lgbm_model.exists():
                        _ = _run_shap("lgbm", lgbm_model, shap_out)
                return True

        # 명령어 구성 (1~4단계)
        cmd = ["python", str(self.script_dir / script)] + args

        self.log_step(f"{step_id}/5: {step_name} 중...")
        self.log_debug(f"실행 명령어: {' '.join(cmd)}")

        # 단계 시작 시간
        start_time = time.time()

        # 스크립트 실행
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.work_dir),  # 작업 디렉토리에서 실행
                capture_output=not self.verbose,  # verbose 모드면 실시간 출력
                text=True,
                check=False,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log_info(f"{script} 완료 (소요: {elapsed:.1f}초)")
                if not self.verbose and result.stdout:
                    # verbose 아닐 때는 마지막 5줄만 출력
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 5:
                        print("...")
                    for line in lines[-5:]:
                        print(line)
                return True
            else:
                self.log_error(f"{script} 실패 (반환 코드: {result.returncode})")
                if result.stderr:
                    print(f"{Colors.RED}=== STDERR ==={Colors.NC}")
                    print(result.stderr)
                if result.stdout:
                    print(f"{Colors.RED}=== STDOUT ==={Colors.NC}")
                    print(result.stdout)
                return False

        except Exception as e:
            self.log_error(f"{script} 실행 중 예외 발생: {e}")
            return False

    def run(self, skip_steps: Optional[List[int]] = None, only_step: Optional[int] = None) -> bool:
        """전체 파이프라인 실행

        Args:
            skip_steps: 스킵할 단계 번호 리스트
            only_step: 특정 단계만 실행

        Returns:
            전체 성공 여부
        """
        skip_steps = skip_steps or []

        # 실행할 단계 필터링
        if only_step:
            steps_to_run = [s for s in self.steps if s["id"] == only_step]
            if not steps_to_run:
                self.log_error(f"단계 {only_step}를 찾을 수 없습니다.")
                return False
        else:
            steps_to_run = [s for s in self.steps if s["id"] not in skip_steps]

        # 시작 로그
        self.log_info(f"RecSys 파이프라인 실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info(f"작업 디렉토리: {self.script_dir}")
        if skip_steps:
            self.log_info(f"스킵할 단계: {skip_steps}")
        if only_step:
            self.log_info(f"실행할 단계: {only_step}")
        print("=" * 60)

        # 전체 시작 시간
        total_start = time.time()

        # 각 단계 실행
        for step in steps_to_run:
            success = self.run_step(step)
            print()  # 단계 구분용 빈 줄

            if not success:
                self.log_error(f"파이프라인 실패: {step['name']} 단계에서 오류 발생")
                return False

        # 완료 로그
        total_elapsed = time.time() - total_start
        minutes = int(total_elapsed // 60)
        seconds = int(total_elapsed % 60)

        print("=" * 60)
        self.log_info(f"모든 파이프라인 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info(f"총 소요 시간: {minutes}분 {seconds}초")

        return True


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="RecSys 파이프라인 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  %(prog)s                    # 전체 파이프라인 실행
  %(prog)s --skip 1,2         # 1,2번 스킵하고 3번부터 실행
  %(prog)s --only 3           # 3번만 실행
  %(prog)s --verbose          # 상세 로그 출력
  %(prog)s --only 5 -v        # 5번만 실행 (상세 모드)
        """
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="스킵할 단계 번호 (쉼표로 구분, 예: 1,2)"
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="특정 단계만 실행 (1~5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력 및 실시간 출력"
    )

    # 학습 단계(5번) 옵션
    parser.add_argument(
        "--train-mode",
        type=str,
        default="holdout",
        choices=["holdout", "cv", "tune", "tune+cv"],
        help="5단계 학습 실행 모드"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="5단계에서 CV 수행 시 fold 수 (기본 5)"
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=30,
        help="튜닝 시 시도 횟수"
    )
    parser.add_argument(
        "--group-by-doctor",
        action="store_true",
        help="CV 시 doctor_id 기준 그룹 분할"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="가능한 경우 GPU 사용"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID"
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 스크립트 디렉토리와 작업 디렉토리 분리
    script_dir = Path(__file__).parent.resolve()  # 스크립트 파일 위치
    work_dir = Path.cwd()  # 현재 작업 디렉토리

    # skip 인자 파싱
    skip_steps = []
    if args.skip:
        try:
            skip_steps = [int(x.strip()) for x in args.skip.split(",")]
        except ValueError:
            print(f"{Colors.RED}[ERROR]{Colors.NC} --skip 인자가 잘못되었습니다: {args.skip}")
            sys.exit(1)

    # 파이프라인 실행
    runner = PipelineRunner(
        script_dir,
        work_dir,
        verbose=args.verbose,
        train_mode=args.train_mode,
        cv_folds=args.cv_folds,
        tune_trials=args.tune_trials,
        group_by_doctor=args.group_by_doctor,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
    )
    success = runner.run(skip_steps=skip_steps, only_step=args.only)

    # 종료 코드
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecSys 파이프라인 통합 실행 스크립트

전체 파이프라인을 순차적으로 실행:
1. Raw Dataset 파싱 (DB → CSV)
2. User Features 처리 (임베딩, 지오코딩 등)
3. Job Features 처리 (급여 계산, 지오코딩 등)
4. Training Table 병합
5. 모델 학습

사용법:
    python run_pipeline.py
    python run_pipeline.py --skip 1,2  # 1,2번 스킵하고 3번부터 실행
    python run_pipeline.py --only 3    # 3번만 실행
    python run_pipeline.py --verbose   # 상세 로그 출력
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class Colors:
    """터미널 색상 정의"""
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class PipelineRunner:
    """파이프라인 실행 관리 클래스"""

    def __init__(self, script_dir: Path, verbose: bool = False):
        self.script_dir = script_dir
        self.verbose = verbose

        # 경로 설정
        self.data_raw_dir = script_dir / "data" / "raw"
        self.data_processed_dir = script_dir / "data" / "processed"
        self.data_training_dir = script_dir / "data" / "training"
        self.models_dir = script_dir / "models"

        # 디렉토리 생성
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 파이프라인 단계 정의 (process/ 디렉토리 하위)
        self.steps = [
            {
                "id": 1,
                "name": "Raw Dataset 파싱",
                "script": "process/01.parse_raw_dataset.py",
                "args": [],
            },
            {
                "id": 2,
                "name": "User Features 처리",
                "script": "process/02.process_user_features.py",
                "args": [
                    "--input", str(self.data_raw_dir / "user_features.csv"),
                    "--out_dir", str(self.data_processed_dir ),
                ],
            },
            {
                "id": 3,
                "name": "Job Features 처리",
                "script": "process/03.process_job_features.py",
                "args": [
                    "--input", str(self.data_raw_dir / "job_features.csv"),
                    "--out_dir", str(self.data_processed_dir ),
                    "--log_interval", "500",
                    "--concurrency", "50",
                ],
            },
            {
                "id": 4,
                "name": "Training Table 병합",
                "script": "process/04.merge_to_training_table.py",
                "args": [
                    "--user_csv", str(self.data_processed_dir / "user_features_processed.csv"),
                    "--job_csv", str(self.data_processed_dir / "job_training_view.csv"),
                    "--out_dir", str(self.data_training_dir),
                ],
            },
            {
                "id": 5,
                "name": "모델 학습",
                "script": "process/05.train_models.py",
                "args": [],  # CSV 경로는 런타임에 결정
            },
        ]

    def log_info(self, msg: str):
        """INFO 로그 출력"""
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

    def log_error(self, msg: str):
        """ERROR 로그 출력"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

    def log_step(self, msg: str):
        """STEP 로그 출력"""
        print(f"{Colors.YELLOW}[STEP]{Colors.NC} {msg}")

    def log_debug(self, msg: str):
        """DEBUG 로그 출력 (verbose 모드일 때만)"""
        if self.verbose:
            print(f"{Colors.CYAN}[DEBUG]{Colors.NC} {msg}")

    def run_step(self, step: dict) -> bool:
        """단일 파이프라인 단계 실행

        Args:
            step: 단계 정의 딕셔너리

        Returns:
            성공 여부 (True/False)
        """
        step_id = step["id"]
        step_name = step["name"]
        script = step["script"]
        args = step["args"].copy()

        # verbose 플래그 추가
        if self.verbose and "--verbose" not in args:
            args.append("--verbose")

        # 5번 단계는 최신 training CSV 찾기
        if step_id == 5:
            training_dir = self.data_training_dir
            training_csvs = sorted(training_dir.glob("training_pairs_*.csv"), reverse=True)
            if not training_csvs:
                self.log_error(f"Training CSV를 찾을 수 없습니다: {training_dir}")
                return False
            training_csv = training_csvs[0]
            # 학습 스크립트 인자 구성 (holdout 기본)
            args = [
                "--input_csv", str(training_csv),
                "--out_dir", str(self.models_dir),
            ] + args
            self.log_debug(f"Training CSV 선택: {training_csv.name}")

        # 명령어 구성
        cmd = ["python", str(self.script_dir / script)] + args

        self.log_step(f"{step_id}/5: {step_name} 중...")
        self.log_debug(f"실행 명령어: {' '.join(cmd)}")

        # 단계 시작 시간
        start_time = time.time()

        # 스크립트 실행
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.script_dir),
                capture_output=not self.verbose,  # verbose 모드면 실시간 출력
                text=True,
                check=False,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log_info(f"{script} 완료 (소요: {elapsed:.1f}초)")
                if not self.verbose and result.stdout:
                    # verbose 아닐 때는 마지막 5줄만 출력
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 5:
                        print("...")
                    for line in lines[-5:]:
                        print(line)
                # 5단계 학습 성공 시 SHAP 계산 시도
                if step_id == 5:
                    def _run_shap(model_type: str, model_path: Path, output_dir: Path) -> bool:
                        shap_script = str(self.script_dir / "tools" / "shap_analysis.py")
                        cmd_shap = [
                            "python", shap_script,
                            "--model_type", model_type,
                            "--model_path", str(model_path),
                            "--input_csv", str(training_csv),
                            "--output_dir", str(output_dir),
                            "--no_plots",
                        ]
                        self.log_debug(f"SHAP 실행 명령어: {' '.join(cmd_shap)}")
                        res = subprocess.run(
                            cmd_shap,
                            cwd=str(self.script_dir),
                            capture_output=not self.verbose,
                            text=True,
                            check=False,
                        )
                        if res.returncode == 0:
                            self.log_info(f"SHAP({model_type}) 계산 완료 → {output_dir}")
                            if not self.verbose and res.stdout:
                                lines2 = res.stdout.strip().split('\n')
                                if len(lines2) > 5:
                                    print("...")
                                for ln in lines2[-5:]:
                                    print(ln)
                            return True
                        else:
                            self.log_error(f"SHAP({model_type}) 계산 실패 (code={res.returncode})")
                            if res.stderr:
                                print(f"{Colors.RED}=== STDERR ==={Colors.NC}")
                                print(res.stderr)
                            if res.stdout:
                                print(f"{Colors.RED}=== STDOUT ==={Colors.NC}")
                                print(res.stdout)
                            return False

                    # holdout 위치 모델 확인
                    shap_out_root = self.models_dir / "shap"
                    shap_out_root.mkdir(parents=True, exist_ok=True)
                    xgb_model = self.models_dir / "xgb_model.json"
                    lgbm_model = self.models_dir / "lgbm_model.txt"
                    if xgb_model.exists():
                        _run_shap("xgb", xgb_model, shap_out_root)
                    if lgbm_model.exists():
                        _run_shap("lgbm", lgbm_model, shap_out_root)

                    # CV 폴드 하위 모델도 있으면 처리
                    for fold_dir in sorted(self.models_dir.glob("fold_*")):
                        shap_out = fold_dir / "shap"
                        shap_out.mkdir(parents=True, exist_ok=True)
                        xgb_fold = fold_dir / "xgb_model.json"
                        lgbm_fold = fold_dir / "lgbm_model.txt"
                        if xgb_fold.exists():
                            _run_shap("xgb", xgb_fold, shap_out)
                        if lgbm_fold.exists():
                            _run_shap("lgbm", lgbm_fold, shap_out)
                return True
            else:
                self.log_error(f"{script} 실패 (반환 코드: {result.returncode})")
                if result.stderr:
                    print(f"{Colors.RED}=== STDERR ==={Colors.NC}")
                    print(result.stderr)
                if result.stdout:
                    print(f"{Colors.RED}=== STDOUT ==={Colors.NC}")
                    print(result.stdout)
                return False

        except Exception as e:
            self.log_error(f"{script} 실행 중 예외 발생: {e}")
            return False

    def run(self, skip_steps: Optional[List[int]] = None, only_step: Optional[int] = None) -> bool:
        """전체 파이프라인 실행

        Args:
            skip_steps: 스킵할 단계 번호 리스트
            only_step: 특정 단계만 실행

        Returns:
            전체 성공 여부
        """
        skip_steps = skip_steps or []

        # 실행할 단계 필터링
        if only_step:
            steps_to_run = [s for s in self.steps if s["id"] == only_step]
            if not steps_to_run:
                self.log_error(f"단계 {only_step}를 찾을 수 없습니다.")
                return False
        else:
            steps_to_run = [s for s in self.steps if s["id"] not in skip_steps]

        # 시작 로그
        self.log_info(f"RecSys 파이프라인 실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info(f"작업 디렉토리: {self.script_dir}")
        if skip_steps:
            self.log_info(f"스킵할 단계: {skip_steps}")
        if only_step:
            self.log_info(f"실행할 단계: {only_step}")
        print("=" * 60)

        # 전체 시작 시간
        total_start = time.time()

        # 각 단계 실행
        for step in steps_to_run:
            success = self.run_step(step)
            print()  # 단계 구분용 빈 줄

            if not success:
                self.log_error(f"파이프라인 실패: {step['name']} 단계에서 오류 발생")
                return False

        # 완료 로그
        total_elapsed = time.time() - total_start
        minutes = int(total_elapsed // 60)
        seconds = int(total_elapsed % 60)

        print("=" * 60)
        self.log_info(f"모든 파이프라인 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info(f"총 소요 시간: {minutes}분 {seconds}초")

        return True


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="RecSys 파이프라인 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  %(prog)s                    # 전체 파이프라인 실행
  %(prog)s --skip 1,2         # 1,2번 스킵하고 3번부터 실행
  %(prog)s --only 3           # 3번만 실행
  %(prog)s --verbose          # 상세 로그 출력
  %(prog)s --only 5 -v        # 5번만 실행 (상세 모드)
        """
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="스킵할 단계 번호 (쉼표로 구분, 예: 1,2)"
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="특정 단계만 실행 (1~5)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로그 출력 및 실시간 출력"
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 스크립트 디렉토리 확인
    script_dir = Path(__file__).parent.resolve()

    # skip 인자 파싱
    skip_steps = []
    if args.skip:
        try:
            skip_steps = [int(x.strip()) for x in args.skip.split(",")]
        except ValueError:
            print(f"{Colors.RED}[ERROR]{Colors.NC} --skip 인자가 잘못되었습니다: {args.skip}")
            sys.exit(1)

    # 파이프라인 실행
    runner = PipelineRunner(script_dir, verbose=args.verbose)
    success = runner.run(skip_steps=skip_steps, only_step=args.only)

    # 종료 코드
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
