"""Fit the preserved final pipeline without replacing frozen reference results."""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from src import aims_training as training
    from scripts.check_release import main as check_release

    check_release(allow_executed_review=True)
    destination = ROOT / "reproduced/final"
    training.ARTIFACT_DIR = destination / "artifacts"
    training.TABLE_DIR = destination / "tables"
    training.FIGURE_DIR = destination / "figures"
    results = training.train_and_evaluate(ROOT / "data/processed/aims_clean_features.csv")
    reference = pd.read_csv(ROOT / "results/reference/final_metrics.csv")
    comparison = results["metrics"].merge(reference, on=["axis", "split", "count"],
                                          suffixes=("_rerun", "_reference"), validate="one_to_one")
    for metric in ("MAE", "RMSE", "R2", "bias", "within_5_percent", "p90_abs_error"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_rerun"] - comparison[f"{metric}_reference"]
    comparison.to_csv(destination / "comparison_to_reference.csv", index=False)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib")},
                   "evaluation_fit_rows": 400, "deployment_fit_rows": 668,
                   "note": "Feature-level reproduction, not raw feature extraction or prospective validation"}
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(comparison[["axis", "split", "MAE_rerun", "MAE_reference", "delta_MAE"]].to_string(index=False))
    print("New artifacts: reproduced/final/ (not committed). Frozen references unchanged.")
    check_release(allow_executed_review=True)


if __name__ == "__main__":
    main()
