import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def load_feature_columns(model_json_path: str, explicit_features_json: Optional[str] = None) -> List[str]:
    """Resolve feature column order.

    Priority:
    1) explicit_features_json if provided
    2) data_info.json placed next to the model json
    3) fallback to importing FEATURE_COLUMNS from scripts.train_models
    """
    if explicit_features_json and os.path.exists(explicit_features_json):
        with open(explicit_features_json, "r") as f:
            data = json.load(f)
        feats = data.get("features")
        if isinstance(feats, list) and feats:
            return [str(c) for c in feats]

    candidate = os.path.join(os.path.dirname(model_json_path), "data_info.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r") as f:
                data = json.load(f)
            feats = data.get("features")
            if isinstance(feats, list) and feats:
                return [str(c) for c in feats]
        except Exception:
            pass

    # No import fallback: enforce explicit features source
    raise RuntimeError(
        "FEATURE_COLUMNS를 찾을 수 없습니다. --features_json를 지정하거나 모델 디렉토리의 data_info.json을 확인하세요."
    )


def load_xgb_booster(model_json_path: str):
    from xgboost import Booster

    booster = Booster()
    booster.load_model(model_json_path)
    return booster


def predict_proba_with_booster(
    booster,
    features_df: pd.DataFrame,
    feature_columns: List[str],
) -> np.ndarray:
    from xgboost import DMatrix

    X = features_df[feature_columns].astype(float).values
    dmat = DMatrix(X)
    # For binary:logistic, this returns P(y=1)
    proba = booster.predict(dmat)
    return np.asarray(proba, dtype=float)


def compute_best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, dict]:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

    # Dense grid for robust search
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1.0
    best_stats = {}
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            cm = confusion_matrix(y_true, y_pred).tolist()
            best_stats = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1),
                "confusion_matrix": cm,
            }
    return best_thr, best_stats


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str,  help="지정 시 해당 파일에 결과 저장, 미지정 시 현재 디렉토리에 inferred.csv 파일 생성")
    parser.add_argument("--threshold", type=float, default=None, help="기본 0.5. 파이프라인에선 별도 루프에서 다중 저장")
    parser.add_argument("--features_json", type=str, default="", help="data_info.json 경로(선택)")

    # Optional: threshold calibration on a labeled CSV having a column named 'applied'
    parser.add_argument("--calibrate_csv", type=str, default="", help="라벨 포함 CSV. 제공 시 F1 최대 임계값 산출")
    args = parser.parse_args()

    feature_columns = load_feature_columns(args.model_json, args.features_json or None)
    booster = load_xgb_booster(args.model_json)

    # Optional threshold calibration
    chosen_threshold: float
    threshold_report: Optional[dict] = None
    if args.threshold is not None:
        chosen_threshold = float(args.threshold)
    elif args.calibrate_csv:
        calib_df = pd.read_csv(args.calibrate_csv)
        if "applied" not in calib_df.columns:
            raise ValueError("--calibrate_csv에는 'applied' 라벨 컬럼이 필요합니다.")
        missing = [c for c in feature_columns if c not in calib_df.columns]
        if missing:
            raise ValueError(f"보정용 CSV에 누락된 피처가 있습니다: {missing}")
        y_true = calib_df["applied"].astype(int).values
        y_prob = predict_proba_with_booster(booster, calib_df, feature_columns)
        thr, stats = compute_best_threshold_by_f1(y_true, y_prob)
        chosen_threshold = float(thr)
        threshold_report = {"threshold": chosen_threshold, **stats}
    else:
        chosen_threshold = 0.5

    # Predictions for the input CSV
    df = pd.read_csv(args.input_csv)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 누락된 피처가 있습니다: {missing}")

    y_prob_input = predict_proba_with_booster(booster, df, feature_columns)
    y_pred_1 = (y_prob_input >= chosen_threshold).astype(int)
    y_pred_0 = 1 - y_pred_1

    # Append columns
    df_out = df.copy()
    df_out["score"] = y_prob_input
    df_out["pred_1"] = y_pred_1
    df_out["pred_0"] = y_pred_0

    # When calibration is not used, if input CSV has labels, compute evaluation metrics at chosen threshold
    if threshold_report is None and "applied" in df.columns:
        y_true_input = df["applied"].astype(int).values
        stats_input = compute_metrics_at_threshold(y_true_input, y_prob_input, float(chosen_threshold))
        threshold_report = {"threshold": float(chosen_threshold), **stats_input}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    # Emit a sidecar JSON with threshold info
    sidecar = {
        "model_json": os.path.abspath(args.model_json),
        "features": feature_columns,
        "threshold": chosen_threshold,
    }
    if threshold_report is not None:
        sidecar["threshold_report"] = threshold_report

    sidecar_path = os.path.splitext(args.output_csv)[0] + "_meta.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_csv": os.path.abspath(args.output_csv), **sidecar}, ensure_ascii=False))


if __name__ == "__main__":
    main()


