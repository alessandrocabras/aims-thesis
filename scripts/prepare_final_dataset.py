"""Generate the final clean AIMS feature dataset and fitted feature transformers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aims_pipeline import build_final_feature_dataset, prepare_samples, save_prepared_dataset


def main() -> None:
    samples = prepare_samples(
        ROOT / "data" / "raw" / "real" / "EffectiveDataset_Tseries_with_azimuth.csv",
        ROOT / "data" / "raw" / "control" / "control_crosscheck_effective_target_projection.csv",
        ROOT / "data" / "raw" / "control" / "control_crosscheck_pointing_audit.csv",
    )
    dataset, artifacts = build_final_feature_dataset(samples)
    report = save_prepared_dataset(
        dataset,
        artifacts,
        samples.skip_counts,
        ROOT / "data" / "processed" / "aims_clean_features.csv",
        ROOT / "artifacts" / "feature_builders.joblib",
        ROOT / "data" / "processed" / "aims_clean_features_report.json",
    )
    print(report)


if __name__ == "__main__":
    main()
