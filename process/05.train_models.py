import argparse
import json
import os
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, train_test_split

# Optional dependency for Bayesian optimization
try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


FEATURE_COLUMNS = [
    "spec_match",
    "distance_home",
    "distance_office",
#    "CAREER_YEARS",
#    "PAY",
    # 경력 매칭 관련 추가 특성
    "career_match",
    "career_gap",
    "is_career_irrelevant",
    "similarity"
]

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "min_child_samples": 20,
    "reg_lambda": 0.0,
    "reg_alpha": 0.0,
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = ["applied", "doctor_id", "board_id"] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure correct dtypes for features and fill missing with 0 (robust to sparse features like similarity)
    for c in FEATURE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Handle label: coerce to numeric, drop only if label is missing
    df["applied"] = pd.to_numeric(df["applied"], errors="coerce")
    df = df.dropna(subset=["applied"]).copy()
    df["applied"] = df["applied"].astype(int)
    return df


def stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split DataFrame; use stratification only when there are at least two classes
    stratify_target = df["applied"] if df["applied"].nunique() >= 2 else None
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_target,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def compute_class_weights(y: np.ndarray) -> float:
    # scale_pos_weight for XGBoost: negative/positive ratio
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return 1.0
    return float(neg / max(pos, 1))


def train_xgboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    seed: int,
    model_params: Dict[str, Any],
) -> Tuple[object, Dict[str, float], pd.DataFrame]:
    from xgboost import XGBClassifier

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["applied"].values
    X_valid = valid_df[FEATURE_COLUMNS].values
    y_valid = valid_df["applied"].values

    base_params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
    )
    if model_params.get("use_gpu", False):
        # XGBoost 2.x: GPU는 device 매개변수만 사용 (gpu_id 금지)
        base_params["device"] = "cuda"
    base_params.update({k: v for k, v in (model_params or {}).items() if k not in ["use_gpu", "gpu_id"]})
    model = XGBClassifier(**base_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
    )

    y_prob = model.predict_proba(X_valid)[:, 1]
    metrics = evaluate_predictions(y_valid, y_prob)

    # Use XGBoost's native feature_importances_
    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_.tolist(),
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, importances


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    seed: int,
    model_params: Dict[str, Any],
) -> Tuple[object, Dict[str, float], pd.DataFrame]:
    from lightgbm import LGBMClassifier

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["applied"].values
    X_valid = valid_df[FEATURE_COLUMNS].values
    y_valid = valid_df["applied"].values

    base_params = dict(
        objective="binary",
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        metric=["binary_logloss", "auc"],
    )
    if model_params.get("use_gpu", False):
        # GPU가 없거나 OpenCL 미설치 환경에서는 CPU로 자동 폴백
        try:
            base_params["device"] = "gpu"
            base_params["gpu_platform_id"] = 0
            base_params["gpu_device_id"] = model_params.get("gpu_id", 0)
        except Exception:
            pass
    base_params.update({k: v for k, v in (model_params or {}).items() if k not in ["use_gpu", "gpu_id"]})
    model = LGBMClassifier(**base_params)

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
        )
    except Exception as fit_err:
        # GPU 환경 미설치(OpenCL 없음 등) 시 CPU로 자동 폴백하여 재시도
        if model_params.get("use_gpu", False):
            base_params_cpu = {k: v for k, v in base_params.items() if k not in ["device", "gpu_platform_id", "gpu_device_id"]}
            base_params_cpu["device"] = "cpu"
            model = LGBMClassifier(**base_params_cpu)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
            )
        else:
            raise

    y_prob = model.predict_proba(X_valid)[:, 1]
    metrics = evaluate_predictions(y_valid, y_prob)

    # Use gain-based feature importance for LightGBM (no fallback)
    gain_importances = model.booster_.feature_importance(importance_type="gain")
    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": gain_importances.tolist(),
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, importances


def train_logistic(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    seed: int,
    model_params: Dict[str, Any] | None = None,
) -> Tuple[object, Dict[str, float], pd.DataFrame]:
    """Train logistic regression and return model, metrics, and coefficients with p-values.

    Preference: statsmodels GLM Binomial to compute p-values. If unavailable, fall back to
    scikit-learn LogisticRegression (p-values will be NaN).
    """
    X_train_df = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["applied"].values
    X_valid_df = valid_df[FEATURE_COLUMNS].copy()
    y_valid = valid_df["applied"].values

    # Try statsmodels first for coefficients and p-values
    try:
        import statsmodels.api as sm  # type: ignore

        X_train_sm = sm.add_constant(X_train_df, has_constant="add")
        X_valid_sm = sm.add_constant(X_valid_df, has_constant="add")

        # Optional weighting for imbalance (weight positives higher)
        # Using simple scheme similar to scale_pos_weight
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        spw = float(neg / max(pos, 1.0)) if pos > 0 else 1.0
        sample_weights = np.ones_like(y_train, dtype=float)
        if spw > 1.0:
            sample_weights[y_train == 1] = spw

        model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial(), freq_weights=sample_weights)
        results = model.fit()

        # Predictions
        y_prob = results.predict(X_valid_sm)
        y_prob = np.asarray(y_prob, dtype=float)

        metrics = evaluate_predictions(y_valid, y_prob)

        # Coefficients and p-values (exclude intercept)
        params = results.params
        pvalues = results.pvalues
        coef_df = (
            pd.DataFrame({
                "feature": params.index,
                "beta": params.values,
                "p_value": pvalues.values,
            })
            .query("feature != 'const'")
            .sort_values("beta", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True)
        )

        return results, metrics, coef_df
    except Exception:
        # Fallback to scikit-learn (no p-values)
        from sklearn.linear_model import LogisticRegression

        X_train = X_train_df.values
        X_valid = X_valid_df.values

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=max(1, os.cpu_count() or 1))
        try:
            clf.fit(X_train, y_train)
        except Exception:
            # Last-resort: try liblinear solver
            clf = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
            clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_valid)[:, 1]
        metrics = evaluate_predictions(y_valid, y_prob)

        coefs = getattr(clf, "coef_", np.zeros((1, len(FEATURE_COLUMNS))))
        coef_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "beta": coefs.flatten().tolist(),
                "p_value": [float("nan")] * len(FEATURE_COLUMNS),
            }
        ).sort_values("beta", key=lambda s: s.abs(), ascending=False)

        return clf, metrics, coef_df


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: Dict[str, float] = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = float("nan")
    metrics["logloss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics


def save_reports(
    out_dir: str,
    model_name: str,
    model_obj: object,
    metrics: Dict[str, float],
    importances: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    # Metrics JSON
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Classification report & confusion matrix
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    with open(os.path.join(out_dir, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(json.dumps(cm))

    # Curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(out_dir, f"{model_name}_roc_curve.csv"), index=False
    )
    pd.DataFrame({"precision": pr_prec, "recall": pr_rec}).to_csv(
        os.path.join(out_dir, f"{model_name}_pr_curve.csv"), index=False
    )

    # Importances / Coefficients
    if model_name == "logi":
        importances.to_csv(
            os.path.join(out_dir, f"{model_name}_coefficients.csv"), index=False
        )
    else:
        importances.to_csv(
            os.path.join(out_dir, f"{model_name}_feature_importances.csv"), index=False
        )

    # Model artifacts
    try:
        from xgboost import XGBClassifier

        if isinstance(model_obj, XGBClassifier):
            model_obj.get_booster().save_model(
                os.path.join(out_dir, f"{model_name}_model.json")
            )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        if isinstance(model_obj, LGBMClassifier):
            model_obj.booster_.save_model(
                os.path.join(out_dir, f"{model_name}_model.txt")
            )
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/SPO/Project/RecSys/data/processed/training/training_pairs_20250929_181817.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/SPO/Project/RecSys/models",
    )
    parser.add_argument("--models", type=str, default="all", choices=["all", "xgb", "lgbm", "logi"])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--top_k_importances", type=int, default=20)
    parser.add_argument("--cv_folds", type=int, default=5, help="0이면 홀드아웃, >0이면 K-fold (기본 5)")
    parser.add_argument("--group_by_doctor", action="store_true", help="doctor_id 기준 그룹 분할")

    # 하이퍼파라미터 CLI 인자 제거: 기본 상수와 JSON 병합 사용

    # GPU settings
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")

    # Bayesian optimization flags
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune_models", type=str, default="all", choices=["all", "xgb", "lgbm"])
    parser.add_argument("--tune_trials", type=int, default=30)

    # Apply tuned params
    parser.add_argument("--xgb_tuning_json", type=str, default="", help="Path to xgb_tuning.json")
    parser.add_argument("--lgbm_tuning_json", type=str, default="", help="Path to lgbm_tuning.json")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = args.out_dir

    def _load_best_params(json_path: str) -> Dict[str, Any]:
        if not json_path:
            return {}
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "best_params" in data:
                return data["best_params"]
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    tuned_xgb = _load_best_params(args.xgb_tuning_json)
    tuned_lgbm = _load_best_params(args.lgbm_tuning_json)

    df = load_dataset(args.input_csv)
    if args.tune:
        if optuna is None:
            raise RuntimeError("optuna가 설치되어 있지 않습니다.")
        run_tuning(df, args, args.out_dir)
        return
    if args.cv_folds and args.cv_folds > 0:
        run_kfold(df, args)
        return
    train_df, valid_df = stratified_split(df, test_size=args.test_size, seed=args.random_seed)

    # Save split sizes
    with open(os.path.join(run_dir, "data_info.json"), "w") as f:
        json.dump(
            {
                "total_rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "features": FEATURE_COLUMNS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    y_valid = valid_df["applied"].values
    X_valid = valid_df[FEATURE_COLUMNS].values

    if args.models in ("all", "xgb"):
        spw = compute_class_weights(train_df["applied"].values)
        xgb_params = dict(DEFAULT_XGB_PARAMS)
        if tuned_xgb:
            allowed_keys = {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "reg_lambda",
                "reg_alpha",
            }
            xgb_params.update({k: v for k, v in tuned_xgb.items() if k in allowed_keys})
        xgb_params.update({
            "scale_pos_weight": spw,
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
        })
        xgb_model, xgb_metrics, xgb_imps = train_xgboost(
            train_df,
            valid_df,
            seed=args.random_seed,
            model_params=xgb_params,
        )
        y_prob = xgb_model.predict_proba(X_valid)[:, 1]
        save_reports(run_dir, "xgb", xgb_model, xgb_metrics, xgb_imps, y_valid, y_prob)

        # Save top-k summary
        xgb_imps.head(args.top_k_importances).to_csv(
            os.path.join(run_dir, "xgb_top_features.csv"), index=False
        )

    if args.models in ("all", "lgbm"):
        lgbm_params = dict(DEFAULT_LGBM_PARAMS)
        if tuned_lgbm:
            allowed_keys = {
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "feature_fraction",
                "bagging_fraction",
                "min_child_samples",
                "reg_lambda",
                "reg_alpha",
            }
            lgbm_params.update({k: v for k, v in tuned_lgbm.items() if k in allowed_keys})
        lgbm_params.update({
            "is_unbalance": True,
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
        })
        lgbm_model, lgbm_metrics, lgbm_imps = train_lightgbm(
            train_df,
            valid_df,
            seed=args.random_seed,
            model_params=lgbm_params,
        )
        y_prob = lgbm_model.predict_proba(X_valid)[:, 1]
        save_reports(run_dir, "lgbm", lgbm_model, lgbm_metrics, lgbm_imps, y_valid, y_prob)
        lgbm_imps.head(args.top_k_importances).to_csv(
            os.path.join(run_dir, "lgbm_top_features.csv"), index=False
        )

    if args.models in ("all", "logi"):
        logi_model, logi_metrics, logi_coefs = train_logistic(
            train_df,
            valid_df,
            seed=args.random_seed,
            model_params={},
        )
        # Predict probabilities for reporting
        try:
            import statsmodels.api as sm  # type: ignore

            X_valid_sm = sm.add_constant(pd.DataFrame(X_valid, columns=FEATURE_COLUMNS), has_constant="add")
            y_prob = np.asarray(logi_model.predict(X_valid_sm), dtype=float)
        except Exception:
            y_prob = logi_model.predict_proba(X_valid)[:, 1]
        save_reports(run_dir, "logi", logi_model, logi_metrics, logi_coefs, y_valid, y_prob)

    print(run_dir)


def run_kfold(df: pd.DataFrame, args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = args.out_dir
    run_dir = args.out_dir

    X = df[FEATURE_COLUMNS]
    y = df["applied"].values

    if y.size == 0:
        raise ValueError(
            "Empty dataset after loading. Check input CSV and feature NaNs (features are now filled with 0)."
        )

    if args.group_by_doctor:
        groups = df["doctor_id"].values
        splitter = GroupKFold(n_splits=args.cv_folds)
        splits = splitter.split(X, y, groups=groups)
    else:
        # If only one class exists, fall back to KFold (non-stratified)
        if np.unique(y).size >= 2:
            splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
            splits = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
            splits = splitter.split(X)

    all_metrics = {"xgb": [], "lgbm": [], "logi": []}
    fold_idx = 0
    for train_idx, valid_idx in splits:
        fold_idx += 1
        fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        # Save train/test datasets for this fold
        try:
            train_path = os.path.join(fold_dir, "train.csv")
            test_path = os.path.join(fold_dir, "test.csv")
            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(test_path, index=False)
        except Exception:
            pass

        with open(os.path.join(fold_dir, "data_info.json"), "w") as f:
            json.dump(
                {
                    "fold": fold_idx,
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "features": FEATURE_COLUMNS,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        y_valid = valid_df["applied"].values
        X_valid = valid_df[FEATURE_COLUMNS].values

        if args.models in ("all", "xgb"):
            spw = compute_class_weights(train_df["applied"].values)
            xgb_params = dict(DEFAULT_XGB_PARAMS)
            xgb_params.update({
                "scale_pos_weight": spw,
                "use_gpu": args.use_gpu,
                "gpu_id": args.gpu_id,
            })
            xgb_model, xgb_metrics, xgb_imps = train_xgboost(
                train_df,
                valid_df,
                seed=args.random_seed,
                model_params=xgb_params,
            )
            y_prob = xgb_model.predict_proba(X_valid)[:, 1]
            save_reports(fold_dir, "xgb", xgb_model, xgb_metrics, xgb_imps, y_valid, y_prob)
            all_metrics["xgb"].append(xgb_metrics)

        if args.models in ("all", "lgbm"):
            lgbm_params = dict(DEFAULT_LGBM_PARAMS)
            lgbm_params.update({
                "is_unbalance": True,
                "use_gpu": args.use_gpu,
                "gpu_id": args.gpu_id,
            })
            lgbm_model, lgbm_metrics, lgbm_imps = train_lightgbm(
                train_df,
                valid_df,
                seed=args.random_seed,
                model_params=lgbm_params,
            )
            y_prob = lgbm_model.predict_proba(X_valid)[:, 1]
            save_reports(fold_dir, "lgbm", lgbm_model, lgbm_metrics, lgbm_imps, y_valid, y_prob)
            all_metrics["lgbm"].append(lgbm_metrics)

        if args.models in ("all", "logi"):
            logi_model, logi_metrics, logi_coefs = train_logistic(
                train_df,
                valid_df,
                seed=args.random_seed,
                model_params={},
            )
            # Predict probabilities for reporting
            try:
                import statsmodels.api as sm  # type: ignore

                X_valid_sm = sm.add_constant(pd.DataFrame(X_valid, columns=FEATURE_COLUMNS), has_constant="add")
                y_prob = np.asarray(logi_model.predict(X_valid_sm), dtype=float)
            except Exception:
                y_prob = logi_model.predict_proba(X_valid)[:, 1]
            save_reports(fold_dir, "logi", logi_model, logi_metrics, logi_coefs, y_valid, y_prob)
            all_metrics["logi"].append(logi_metrics)

    # Aggregate metrics (mean and std), overwrite summary
    summary = {}
    for model_name, metrics_list in all_metrics.items():
        if not metrics_list:
            continue
        keys = metrics_list[0].keys()
        mean_dict = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
        std_dict = {k: float(np.std([m[k] for m in metrics_list])) for k in keys}
        summary[model_name] = {"mean": mean_dict, "std": std_dict, "folds": metrics_list}
    with open(os.path.join(run_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(run_dir)


def run_tuning(df: pd.DataFrame, args: argparse.Namespace, base_out: str) -> None:

    os.makedirs(base_out, exist_ok=True)
    run_dir = base_out

    # If cv_folds <= 0, perform holdout tuning with args.test_size (no CV)
    use_holdout = args.cv_folds is None or int(args.cv_folds) <= 0
    if not use_holdout:
        if args.group_by_doctor:
            groups = df["doctor_id"].values
            splitter = GroupKFold(n_splits=args.cv_folds)
            splits = list(splitter.split(df[FEATURE_COLUMNS], df["applied"].values, groups=groups))
        else:
            splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
            splits = list(splitter.split(df[FEATURE_COLUMNS], df["applied"].values))

    def objective_xgb(trial: "optuna.trial.Trial") -> float:
        # Suggest params
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
        }
        if use_holdout:
            train_df, valid_df = stratified_split(df, test_size=args.test_size, seed=args.random_seed)
            spw = compute_class_weights(train_df["applied"].values)
            params_with_spw = dict(params)
            params_with_spw["scale_pos_weight"] = spw
            _model, metrics, _ = train_xgboost(train_df, valid_df, args.random_seed, params_with_spw)
            return float(metrics.get("roc_auc", 0.0))
        aucs = []
        for train_idx, valid_idx in splits:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            spw = compute_class_weights(train_df["applied"].values)
            params_with_spw = dict(params)
            params_with_spw["scale_pos_weight"] = spw
            _model, metrics, _ = train_xgboost(train_df, valid_df, args.random_seed, params_with_spw)
            aucs.append(metrics["roc_auc"])
        return float(np.mean(aucs))

    def objective_lgbm(trial: "optuna.trial.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "is_unbalance": True,
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
        }
        if use_holdout:
            train_df, valid_df = stratified_split(df, test_size=args.test_size, seed=args.random_seed)
            _model, metrics, _ = train_lightgbm(train_df, valid_df, args.random_seed, params)
            return float(metrics.get("roc_auc", 0.0))
        aucs = []
        for train_idx, valid_idx in splits:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)
            _model, metrics, _ = train_lightgbm(train_df, valid_df, args.random_seed, params)
            aucs.append(metrics["roc_auc"])
        return float(np.mean(aucs))

    results = {}
    if args.tune_models in ("all", "xgb"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_xgb, n_trials=args.tune_trials, show_progress_bar=False)
        results["xgb"] = {"best_params": study.best_params, "best_value": study.best_value}
        with open(os.path.join(run_dir, "xgb_tuning.json"), "w") as f:
            json.dump(results["xgb"], f, ensure_ascii=False, indent=2)

    if args.tune_models in ("all", "lgbm"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_lgbm, n_trials=args.tune_trials, show_progress_bar=False)
        results["lgbm"] = {"best_params": study.best_params, "best_value": study.best_value}
        with open(os.path.join(run_dir, "lgbm_tuning.json"), "w") as f:
            json.dump(results["lgbm"], f, ensure_ascii=False, indent=2)

    print(run_dir)


if __name__ == "__main__":
    main()


