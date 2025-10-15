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
            args = [
                "--csv", str(training_csv),
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
