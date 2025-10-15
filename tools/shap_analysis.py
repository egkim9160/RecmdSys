import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def load_feature_columns(model_path: str, explicit_features_json: Optional[str] = None) -> List[str]:
    """Resolve feature column order for SHAP computation.

    Priority:
    1) explicit_features_json if provided
    2) data_info.json placed next to the model file
    3) fallback to importing FEATURE_COLUMNS from scripts.05.train_models
    """
    if explicit_features_json and os.path.exists(explicit_features_json):
        with open(explicit_features_json, "r") as f:
            data = json.load(f)
        feats = data.get("features")
        if isinstance(feats, list) and feats:
            return [str(c) for c in feats]

    candidate = os.path.join(os.path.dirname(model_path), "data_info.json")
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


def compute_shap_xgb(model_path: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from xgboost import Booster, DMatrix

    booster = Booster()
    booster.load_model(model_path)
    dmat = DMatrix(X)
    contribs = booster.predict(dmat, pred_contribs=True)
    # contribs includes bias term as last column
    shap_sum = np.sum(contribs, axis=1)
    prob_from_shap = 1.0 / (1.0 + np.exp(-shap_sum))
    return contribs, prob_from_shap


def compute_shap_lgbm(model_path: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=model_path)
    contribs = booster.predict(X, pred_contrib=True)
    # contribs includes bias term as last column
    shap_sum = np.sum(contribs, axis=1)
    prob_from_shap = 1.0 / (1.0 + np.exp(-shap_sum))
    return contribs, prob_from_shap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["xgb", "lgbm"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--features_json", type=str, default="")
    parser.add_argument("--sample_n", type=int, default=0, help="0이면 전체, >0이면 상한 샘플 수")
    parser.add_argument("--no_plots", action="store_true", help="플롯 생성을 비활성화")
    parser.add_argument("--show_plots", action="store_true", help="가능하면 화면에 플롯 표시(헤드리스 환경에선 실패 가능)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    feature_columns = load_feature_columns(args.model_path, args.features_json or None)
    df = pd.read_csv(args.input_csv)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 누락된 피처가 있습니다: {missing}")

    if args.sample_n and args.sample_n > 0 and len(df) > args.sample_n:
        df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)

    X = df[feature_columns].astype(float).values

    if args.model_type == "xgb":
        contribs, prob_from_shap = compute_shap_xgb(args.model_path, X)
    else:
        contribs, prob_from_shap = compute_shap_lgbm(args.model_path, X)

    # Build DataFrame of SHAP values
    shap_cols = [f"shap_{c}" for c in feature_columns] + ["shap_bias"]
    shap_df = pd.DataFrame(contribs, columns=shap_cols)
    shap_df["shap_sum"] = shap_df[[c for c in shap_cols]].sum(axis=1)
    shap_df["shap_prob"] = prob_from_shap

    # Save SHAP values (could be large)
    shap_values_path = os.path.join(args.output_dir, f"{args.model_type}_shap_values.csv")
    shap_df.to_csv(shap_values_path, index=False)

    # Summary (global) statistics
    summary_rows = []
    for c in feature_columns:
        v = shap_df[f"shap_{c}"].values
        summary_rows.append(
            {
                "feature": c,
                "mean_abs_shap": float(np.mean(np.abs(v))),
                "mean_shap": float(np.mean(v)),
                "std_shap": float(np.std(v)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_shap", ascending=False)
    summary_path = os.path.join(args.output_dir, f"{args.model_type}_shap_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Specified feature directional plot (optional simple bar) and overall bar plot
    if not args.no_plots:
        try:
            import matplotlib
            # Use Agg for headless unless explicitly asked to show
            if not args.show_plots:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Optional SHAP beeswarm summary plot using shap library
            try:
                import shap  # type: ignore
                shap_values_for_plot = contribs[:, : len(feature_columns)]  # drop bias
                plt.figure(figsize=(9, max(4, 0.35 * len(feature_columns) + 2)))
                shap.summary_plot(
                    shap_values_for_plot,
                    X,
                    feature_names=feature_columns,
                    show=False,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_shap_beeswarm.png"), dpi=160)
                plt.close()
            except Exception:
                pass

            # Global bar plot by mean_abs_shap
            top_k = min(20, len(summary_df))
            plt.figure(figsize=(8, 0.35 * top_k + 1.5))
            tmp = summary_df.head(top_k)[::-1]
            plt.barh(tmp["feature"], tmp["mean_abs_shap"], color="#2c7fb8")
            plt.xlabel("mean(|SHAP|)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_shap_bar.png"), dpi=160)
            plt.close()

            # If spec_match exists, show its directional distribution
            if "spec_match" in feature_columns:
                vals = shap_df["shap_spec_match"].values
                plt.figure(figsize=(6, 3))
                plt.hist(vals, bins=40, color="#41ab5d", alpha=0.85)
                plt.title("SHAP(spec_match)")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{args.model_type}_shap_spec_match_hist.png"), dpi=160)
                plt.close()
            if args.show_plots:
                try:
                    plt.show()
                except Exception:
                    pass
        except Exception:
            pass

    # Emit meta
    meta = {
        "model_type": args.model_type,
        "model_path": os.path.abspath(args.model_path),
        "input_csv": os.path.abspath(args.input_csv),
        "output_dir": os.path.abspath(args.output_dir),
        "features": feature_columns,
        "shap_values_csv": os.path.abspath(shap_values_path),
        "shap_summary_csv": os.path.abspath(summary_path),
    }
    with open(os.path.join(args.output_dir, f"{args.model_type}_shap_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()


