import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import argparse


def detect_label_from_filename(path: Path) -> str:
    name = path.name
    if "calibration" in name:
        return "thr_calib"
    try:
        after_prefix = name.split("inferenced_")[1]
        thr_str = after_prefix.split(".output_meta.json")[0]
        return f"thr_{thr_str}"
    except Exception:
        return name


def read_threshold_report(meta_path: Path) -> Dict[str, Any]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    rep = data.get("threshold_report", {})
    return {
        "threshold": rep.get("threshold", data.get("threshold")),
        "accuracy": rep.get("accuracy"),
        "precision": rep.get("precision"),
        "recall": rep.get("recall"),
        "f1": rep.get("f1"),
    }


def write_csv(metrics: Dict[str, Dict[str, float]], out_csv: Path) -> None:
    cols = ["metric", *sorted(metrics.keys(), key=lambda k: (k != "thr_calib", k))]
    rows: List[Dict[str, Any]] = []
    for m in ["accuracy", "precision", "recall", "f1"]:
        row = {"metric": m}
        for c in cols[1:]:
            v = metrics.get(c, {}).get(m)
            row[c] = None if v is None else round(float(v), 4)
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate infer meta JSONs into a CSV table")
    p.add_argument("--meta_dir", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()

    meta_dir = Path(args.meta_dir)
    metas = sorted(meta_dir.glob("inferenced_*.output_meta.json"))
    metas += sorted(meta_dir.glob("inferenced_*calibration*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No meta json under {meta_dir}")
    metrics: Dict[str, Dict[str, float]] = {}
    for m in metas:
        label = detect_label_from_filename(m)
        metrics[label] = read_threshold_report(m)
    write_csv(metrics, Path(args.out_csv))


if __name__ == "__main__":
    main()


