"""Verify published evidence using only the Python standard library."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import ast
import csv
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, context, tolerance=1e-7):
    require(math.isfinite(actual) and abs(actual - expected) <= tolerance,
            f"{context}: {actual} != {expected}")


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    left = int(index)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def metrics(records):
    target = [float(r["target"]) for r in records]
    error = [t - float(r["prediction"]) for t, r in zip(target, records)]
    absolute = [abs(e) for e in error]
    sse = sum(e * e for e in error)
    mean = statistics.mean(target)
    sst = sum((t - mean) ** 2 for t in target)
    return {"count": len(records), "MAE": statistics.mean(absolute),
            "RMSE": math.sqrt(sse / len(records)), "R2": 1 - sse / sst if sst else 0,
            "bias": statistics.mean(error), "p90_abs_error": quantile(absolute, .9),
            "within_5_percent": 100 * sum(e <= 5 for e in absolute) / len(records)}


def validate_dataset(data):
    require(len(data) == 668, "Expected 668 events")
    features = [k for k in data[0] if k.startswith("feature__")]
    require(len(features) == 43 and len(data[0]) == 51, "Feature schema changed")
    times = [datetime.fromisoformat(r["Timestamp"]) for r in data]
    require(len(set(times)) == 668 and times == sorted(times), "Timestamps not unique/ordered")
    require([r["Split"] for r in data] == ["train"] * 400 + ["validation"] * 133 + ["test"] * 135,
            "Chronological split changed")
    for row in data:
        for key in features + ["Elevation", "Azimuth", "target_XEL_arcsec", "target_EL_arcsec"]:
            require(math.isfinite(float(row[key])), f"Nonfinite value: {key}")
    return {r["Timestamp"]: r for r in data}


def validate_predictions(data, predictions):
    lookup = validate_dataset(data)
    expected = {(r["Timestamp"], axis) for r in data if r["Split"] != "train" for axis in ("XEL", "EL")}
    keys = [(r["Timestamp"], r["axis"]) for r in predictions]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Prediction population mismatch")
    for row in predictions:
        original = lookup[row["Timestamp"]]
        for key in ("ReceiverCode", "ReceiverBand"):
            require(row[key] == original[key], f"Prediction metadata mismatch: {key}")
        for key in ("Elevation", "Azimuth"):
            close(float(row[key]), float(original[key]), key)
        target, pred = float(row["target"]), float(row["prediction"])
        close(target, float(original[f"target_{row['axis']}_arcsec"]), "Prediction target")
        close(float(row["residual"]), target - pred, "Residual")
        close(float(row["abs_error"]), abs(target - pred), "Absolute error")
    return lookup


def validate_tables(root, data, predictions):
    lookup = validate_predictions(data, predictions)
    for name in ("final_metrics.csv", "final_metrics_by_receiver.csv", "final_metrics_by_elevation.csv"):
        for summary in rows(root / "results/reference" / name):
            subset = [r for r in predictions if r["axis"] == summary["axis"]
                      and lookup[r["Timestamp"]]["Split"] == summary.get("split", "test")]
            if "ReceiverBand" in summary:
                subset = [r for r in subset if r["ReceiverBand"] == summary["ReceiverBand"]]
            if "elevation_band" in summary:
                lower, upper = map(float, summary["elevation_band"][1:-1].split(","))
                subset = [r for r in subset if lower < float(r["Elevation"]) <= upper]
            require(bool(subset), f"Empty metric population: {summary}")
            for key, value in metrics(subset).items():
                if key in summary:
                    close(value, float(summary[key]), f"{name}: {summary['axis']} {key}")
    baselines = {}
    for axis in ("XEL", "EL"):
        column = f"target_{axis}_arcsec"
        median = statistics.median(float(r[column]) for r in data if r["Split"] == "train")
        mae = statistics.mean(abs(float(r[column]) - median) for r in data if r["Split"] == "test")
        baselines[axis] = {"train_median": median, "historical_test_MAE": mae}
    close(baselines["XEL"]["historical_test_MAE"], 5.077448762, "XEL median baseline")
    close(baselines["EL"]["historical_test_MAE"], 13.171299790, "EL median baseline")
    return baselines


def validate_hashes(root):
    manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = root / item["path"]
        require(path.is_file(), f"Missing released file: {item['path']}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == item["release_sha256"], f"Release hash mismatch: {item['path']}")
    source = root / "thesis_figures/source_data"
    snapshots = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in snapshots["files"].items():
        require(hashlib.sha256((source / name).read_bytes()).hexdigest() == expected,
                f"Plot snapshot hash mismatch: {name}")
    return len(manifest["files"])


def validate_public_tree(root, allow_executed_review=False):
    ignored = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "reproduced", "outputs", "qa"}
    forbidden = {".dat", ".log", ".xlsx", ".xls", ".joblib", ".pkl", ".h5", ".keras", ".npz", ".npy"}
    count = 0
    # High-confidence patterns only; this is not a comprehensive secret scanner.
    secret = re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")
    retired_name = re.compile("met" + "is|ai" + "met", re.IGNORECASE)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in ignored for part in relative.parts):
            continue
        require(not retired_name.search(relative.as_posix()), f"Retired system name in path: {relative}")
        require(path.suffix.lower() not in forbidden, f"Restricted file: {relative}")
        require(not path.name.startswith(".env"), f"Environment file: {relative}")
        require(path.stat().st_size < 10_000_000, f"Oversized file: {relative}")
        if path.suffix in {".py", ".md", ".json", ".csv", ".ipynb", ".yml"}:
            text = path.read_text(encoding="utf-8-sig")
            require(not retired_name.search(text), f"Retired system name in text: {relative}")
            require(not secret.search(text), f"Possible credential: {relative}")
            require(not re.search(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z]", text),
                    f"Personal absolute path: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    executed_review = allow_executed_review and relative.parts[:2] == ("notebooks", "review")
                    require(executed_review or (not cell.get("outputs") and cell.get("execution_count") is None),
                            f"Embedded notebook output: {relative} cell {i}")
                    code = "".join(cell["source"])
                    # Notebook shell/magic lines are IPython, not Python syntax.
                    code = "\n".join("pass" if line.lstrip().startswith(("%", "!")) else line
                                     for line in code.splitlines())
                    ast.parse(code, filename=f"{relative}:cell{i}")
            count += 1
    return count


def main(allow_executed_review=False):
    data = rows(ROOT / "data/processed/aims_clean_features.csv")
    predictions = rows(ROOT / "results/reference/final_predictions.csv")
    baselines = validate_tables(ROOT, data, predictions)
    files = validate_hashes(ROOT)
    notebooks = validate_public_tree(ROOT, allow_executed_review=allow_executed_review)
    print(f"PASS: {len(data)} events, 43 features, 400/133/135 split; {len(predictions)} predictions")
    print(f"PASS: global, receiver and elevation metrics; {files} provenance hashes; {notebooks} notebook syntax checks")
    if allow_executed_review:
        print("Local review-notebook outputs allowed. Run default checks on a clean copy before publishing.")
    print(json.dumps({"training_median_baselines": baselines}, indent=2))
    print("Historical retrospective results only. Restricted-input experiments were not rerun.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-executed-review", action="store_true",
                        help="Allow saved outputs only in notebooks/review during an interactive local run")
    main(allow_executed_review=parser.parse_args().allow_executed_review)
