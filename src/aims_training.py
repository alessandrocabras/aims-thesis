"""Train, evaluate and serialize the final AIMS XEL and EL models."""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "aims_clean_features.csv"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
SPEC_ARCSEC = 5.0


def xel_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=3000)),
    ])


def el_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])


def regression_metrics(axis: str, split: str, target: np.ndarray, prediction: np.ndarray) -> dict:
    error = target - prediction
    absolute = np.abs(error)
    return {
        "axis": axis,
        "split": split,
        "count": len(target),
        "MAE": float(absolute.mean()),
        "RMSE": float(np.sqrt(mean_squared_error(target, prediction))),
        "R2": float(r2_score(target, prediction)),
        "bias": float(error.mean()),
        "within_5_percent": float((absolute <= SPEC_ARCSEC).mean() * 100),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
    }


def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a raster preview and a vector version for the thesis."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_thesis_plots(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics_by_receiver: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"XEL": "#0072B2", "EL": "#D55E00"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    split_colors = {"train": "#0072B2", "validation": "#E69F00", "test": "#D55E00"}
    for axis_name, target_column, ax in [
        ("XEL", "target_XEL_arcsec", axes[0]),
        ("EL", "target_EL_arcsec", axes[1]),
    ]:
        for split_name in ["train", "validation", "test"]:
            subset = data[data["Split"].eq(split_name)]
            ax.scatter(
                subset["Timestamp"], subset[target_column], s=14, alpha=0.65,
                color=split_colors[split_name], label=split_name.capitalize(),
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"{axis_name} [arcsec]")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Quality-controlled operational dataset and chronological split")
    axes[0].legend(ncol=3)
    axes[1].set_xlabel("Time [UTC]")
    fig.tight_layout()
    _save_figure(fig, "00_dataset_timeline_and_splits")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis_name, ax in zip(["XEL", "EL"], axes):
        subset = predictions[predictions["axis"] == axis_name]
        ax.plot(subset["Timestamp"].to_numpy(), subset["target"].to_numpy(), ".", ms=4, alpha=0.65, label="Observed")
        ax.plot(subset["Timestamp"].to_numpy(), subset["prediction"].to_numpy(), ".", ms=4, alpha=0.65, label="Predicted")
        _style_axis(ax, f"{axis_name}: observations and predictions in the temporal test set", "", "Offset [arcsec]")
        ax.legend(ncol=2)
    axes[-1].set_xlabel("Time [UTC]")
    fig.tight_layout()
    _save_figure(fig, "01_test_predictions_timeline")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis_name, ax in zip(["XEL", "EL"], axes):
        subset = predictions[predictions["axis"] == axis_name]
        limit = float(np.max(np.abs(np.r_[subset["target"], subset["prediction"]])))
        ax.scatter(subset["target"].to_numpy(), subset["prediction"].to_numpy(), s=22, alpha=0.65, color=colors[axis_name])
        ax.plot([-limit, limit], [-limit, limit], "k--", lw=1)
        _style_axis(ax, axis_name, "Observed offset [arcsec]", "Predicted offset [arcsec]")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save_figure(fig, "02_observed_vs_predicted")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis_name, ax in zip(["XEL", "EL"], axes):
        subset = predictions[predictions["axis"] == axis_name]
        scatter = ax.scatter(
            subset["Elevation"].to_numpy(), subset["residual"].to_numpy(),
            c=subset["abs_error"].to_numpy(), cmap="viridis", s=24, alpha=0.75,
        )
        ax.axhline(0, color="black", lw=1)
        ax.axhline(SPEC_ARCSEC, color="#009E73", ls="--", lw=1)
        ax.axhline(-SPEC_ARCSEC, color="#009E73", ls="--", lw=1)
        _style_axis(ax, f"{axis_name} residuals by elevation", "Elevation [deg]", "Signed residual [arcsec]")
        fig.colorbar(scatter, ax=ax, label="Absolute error [arcsec]")
    fig.tight_layout()
    _save_figure(fig, "03_residuals_by_elevation")

    pivot = metrics_by_receiver.pivot(index="ReceiverBand", columns="axis", values="MAE").reindex(["K", "C_high", "C_low", "Q"]).dropna(how="all")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, color=[colors.get(c, "gray") for c in pivot.columns])
    ax.axhline(SPEC_ARCSEC, color="#009E73", ls="--", label="5-arcsec benchmark")
    _style_axis(ax, "Test-set MAE by receiver band", "Receiver band", "MAE [arcsec]")
    ax.tick_params(axis="x", rotation=0)
    for container, axis_name in zip(ax.containers, pivot.columns):
        counts = metrics_by_receiver.set_index(["ReceiverBand", "axis"])["count"]
        labels = [f"{bar.get_height():.2f}\n(n={int(counts.get((band, axis_name), 0))})" for bar, band in zip(container, pivot.index)]
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)
    ax.legend()
    ax.margins(y=0.16)
    fig.tight_layout()
    _save_figure(fig, "04_mae_by_receiver")

    fig, (ax, tail_ax) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    for axis_name in ["XEL", "EL"]:
        values = np.sort(predictions.loc[predictions["axis"] == axis_name, "abs_error"].to_numpy())
        probability = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, probability * 100, lw=2, color=colors[axis_name], label=axis_name)
        tail_ax.plot(values, probability * 100, lw=2, color=colors[axis_name], label=axis_name)
    ax.axvline(SPEC_ARCSEC, color="#009E73", ls="--", label="5-arcsec benchmark")
    tail_ax.axvline(SPEC_ARCSEC, color="#009E73", ls="--")
    _style_axis(ax, "Absolute-error empirical CDF", "Absolute error [arcsec]", "Cumulative samples [%]")
    _style_axis(tail_ax, "Full residual tail", "Absolute error [arcsec]", "Cumulative samples [%]")
    ax.set_xlim(left=0, right=25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "05_absolute_error_ecdf")


def train_and_evaluate(dataset_path: Path = DATASET_PATH) -> dict:
    data = pd.read_csv(dataset_path, parse_dates=["Timestamp"])
    feature_columns = [c for c in data.columns if c.startswith("feature__")]
    train = data["Split"].eq("train")
    validation = data["Split"].eq("validation")
    test = data["Split"].eq("test")

    xel = xel_model().fit(data.loc[train, feature_columns], data.loc[train, "target_XEL_arcsec"])
    el = el_model().fit(data.loc[train, feature_columns], data.loc[train, "target_EL_arcsec"])

    metrics = []
    prediction_frames = []
    for split_name, mask in [("validation", validation), ("test", test)]:
        for axis, model, features, target_col in [
            ("XEL", xel, feature_columns, "target_XEL_arcsec"),
            ("EL", el, feature_columns, "target_EL_arcsec"),
        ]:
            target = data.loc[mask, target_col].to_numpy()
            prediction = model.predict(data.loc[mask, features])
            metrics.append(regression_metrics(axis, split_name, target, prediction))
            frame = data.loc[mask, ["Timestamp", "Elevation", "Azimuth", "ReceiverCode", "ReceiverBand"]].copy()
            frame["axis"] = axis
            frame["target"] = target
            frame["prediction"] = prediction
            frame["residual"] = target - prediction
            frame["abs_error"] = np.abs(frame["residual"])
            prediction_frames.append(frame)

    metrics_df = pd.DataFrame(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    test_predictions = predictions[predictions["Timestamp"].isin(data.loc[test, "Timestamp"])]
    receiver_metrics = test_predictions.groupby(["axis", "ReceiverBand"], observed=True).agg(
        count=("abs_error", "size"),
        MAE=("abs_error", "mean"),
        bias=("residual", "mean"),
        within_5_percent=("abs_error", lambda values: (values <= SPEC_ARCSEC).mean() * 100),
        p90_abs_error=("abs_error", lambda values: values.quantile(0.90)),
    ).reset_index()
    elevation_metrics = test_predictions.assign(
        elevation_band=lambda frame: pd.cut(frame["Elevation"], [0, 20, 30, 45, 60, 75, 90], include_lowest=True)
    ).groupby(["axis", "elevation_band"], observed=True).agg(
        count=("abs_error", "size"), MAE=("abs_error", "mean"),
        within_5_percent=("abs_error", lambda values: (values <= SPEC_ARCSEC).mean() * 100),
        p90_abs_error=("abs_error", lambda values: values.quantile(0.90)),
    ).reset_index()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "xel_model": xel,
        "el_model": el,
        "feature_columns": feature_columns,
    }, ARTIFACT_DIR / "evaluation_models.joblib")
    metrics_df.to_csv(TABLE_DIR / "final_metrics.csv", index=False)
    receiver_metrics.to_csv(TABLE_DIR / "final_metrics_by_receiver.csv", index=False)
    elevation_metrics.to_csv(TABLE_DIR / "final_metrics_by_elevation.csv", index=False)
    predictions.to_csv(TABLE_DIR / "final_predictions.csv", index=False)
    save_thesis_plots(data, test_predictions, receiver_metrics)

    # All-event deployment refits are separate from the reported 400-event evaluation fits.
    deployment_xel = xel_model().fit(data[feature_columns], data["target_XEL_arcsec"])
    deployment_el = el_model().fit(data[feature_columns], data["target_EL_arcsec"])
    joblib.dump({
        "xel_model": deployment_xel,
        "el_model": deployment_el,
        "feature_columns": feature_columns,
        "trained_rows": len(data),
    }, ARTIFACT_DIR / "deployment_models.joblib")
    return {
        "metrics": metrics_df,
        "receiver_metrics": receiver_metrics,
        "elevation_metrics": elevation_metrics,
        "predictions": predictions,
    }


def main() -> None:
    results = train_and_evaluate()
    print(results["metrics"].to_string(index=False))
    print("\nReceiver metrics")
    print(results["receiver_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
