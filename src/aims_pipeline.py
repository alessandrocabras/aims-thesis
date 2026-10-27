"""Final data preparation and feature engineering for SRT AIMS models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
SEQ_LEN = 60
STEP_SECONDS = 120
MAX_SENSOR_RATE_C = 5.0
RECEIVER_LEVELS = ["K", "C_high", "C_low", "Q", "unknown"]


def normalize_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        match = re.match(r"^Sensors_\s*(\d+)$", str(col))
        if match:
            mapping[col] = f"Sensors_{int(match.group(1)):03d}"
    return df.rename(columns=mapping)


def sensor_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^Sensors_\d{3}$", str(c))]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def receiver_one_hot(meta: pd.DataFrame) -> pd.DataFrame:
    band = meta["ReceiverBand"].fillna("unknown").astype(str)
    category = pd.Categorical(band, categories=RECEIVER_LEVELS)
    return pd.get_dummies(category, prefix="rx", dtype=float).reset_index(drop=True)


def chronological_split(n: int, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> np.ndarray:
    n_train = int(n * train_fraction)
    n_validation = int(n * validation_fraction)
    labels = np.full(n, "test", dtype=object)
    labels[:n_train] = "train"
    labels[n_train:n_train + n_validation] = "validation"
    return labels


def scalar_physical_features(x: np.ndarray, elevation: np.ndarray) -> pd.DataFrame:
    current = x[:, -1, :]
    start = x[:, 0, :]
    sensor_std_time = x.std(axis=1)
    delta = np.diff(x, axis=1)
    trend = current - start
    elev = elevation.reshape(-1)
    return pd.DataFrame({
        "elevation": elev,
        "sin_el": np.sin(np.deg2rad(elev)),
        "cos_el": np.cos(np.deg2rad(elev)),
        "current_mean": current.mean(axis=1),
        "current_std_spatial": current.std(axis=1),
        "current_min": current.min(axis=1),
        "current_max": current.max(axis=1),
        "current_range": current.max(axis=1) - current.min(axis=1),
        "window_mean": x.mean(axis=(1, 2)),
        "window_std": x.std(axis=(1, 2)),
        "mean_sensor_std_time": sensor_std_time.mean(axis=1),
        "trend_mean": trend.mean(axis=1),
        "trend_std": trend.std(axis=1),
        "trend_range": trend.max(axis=1) - trend.min(axis=1),
        "abs_rate_mean": np.abs(delta).mean(axis=(1, 2)),
        "abs_rate_max": np.abs(delta).max(axis=(1, 2)),
    })


def temporal_context_features(meta: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(meta["Timestamp"])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0 + timestamp.dt.second / 3600.0
    day = timestamp.dt.dayofyear.astype(float)
    azimuth = pd.to_numeric(meta["Azimuth"], errors="coerce")
    azimuth = azimuth.fillna(azimuth.median())
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "doy_sin": np.sin(2 * np.pi * day / 365.25),
        "doy_cos": np.cos(2 * np.pi * day / 365.25),
        "az_sin": np.sin(np.deg2rad(azimuth)),
        "az_cos": np.cos(np.deg2rad(azimuth)),
    }).reset_index(drop=True)


class ThermalFeatureBuilder:
    """Current-state and two-hour trend PCA shared by the final XEL and EL branches."""

    def __init__(self, n_current: int = 8, n_trend: int = 8):
        self.n_current = n_current
        self.n_trend = n_trend
        self.current_scaler = StandardScaler()
        self.trend_scaler = StandardScaler()
        self.current_pca = PCA(n_components=n_current, random_state=RANDOM_SEED)
        self.trend_pca = PCA(n_components=n_trend, random_state=RANDOM_SEED)

    def fit(self, x: np.ndarray) -> "ThermalFeatureBuilder":
        current = x[:, -1, :]
        trend = current - x[:, 0, :]
        self.current_pca.fit(self.current_scaler.fit_transform(current))
        self.trend_pca.fit(self.trend_scaler.fit_transform(trend))
        return self

    def transform(self, x: np.ndarray, elevation: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        output = pd.concat([
            scalar_physical_features(x, elevation),
            temporal_context_features(meta),
        ], axis=1)
        current = self.current_pca.transform(self.current_scaler.transform(x[:, -1, :]))
        trend_values = x[:, -1, :] - x[:, 0, :]
        trend = self.trend_pca.transform(self.trend_scaler.transform(trend_values))
        for index in range(current.shape[1]):
            output[f"pc_current_{index + 1:02d}"] = current[:, index]
        for index in range(trend.shape[1]):
            output[f"pc_trend_{index + 1:02d}"] = trend[:, index]
        return output


class ELFeatureBuilder:
    """Multi-scale state, centered pattern and trend PCA used by the final EL branch."""

    VIEW_NAMES = ["current", "centered", "trend_30m", "trend_60m", "trend_120m"]

    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.scalers = {name: StandardScaler() for name in self.VIEW_NAMES}
        self.pcas = {
            name: PCA(n_components=n_components, random_state=RANDOM_SEED)
            for name in self.VIEW_NAMES
        }

    @staticmethod
    def views(x: np.ndarray) -> dict[str, np.ndarray]:
        current = x[:, -1, :]
        lag_30 = x[:, max(0, x.shape[1] - 16), :]
        lag_60 = x[:, max(0, x.shape[1] - 31), :]
        return {
            "current": current,
            "centered": current - current.mean(axis=1, keepdims=True),
            "trend_30m": current - lag_30,
            "trend_60m": current - lag_60,
            "trend_120m": current - x[:, 0, :],
        }

    def fit(self, x: np.ndarray) -> "ELFeatureBuilder":
        for name, values in self.views(x).items():
            self.pcas[name].fit(self.scalers[name].fit_transform(values))
        return self

    def transform(self, x: np.ndarray, elevation: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
        output = pd.concat([
            scalar_physical_features(x, elevation),
            temporal_context_features(meta),
        ], axis=1)
        for name, values in self.views(x).items():
            components = self.pcas[name].transform(self.scalers[name].transform(values))
            for index in range(components.shape[1]):
                output[f"ms_{name}_{index + 1:02d}"] = components[:, index]
        current = x[:, -1, :]
        recent_start = x[:, -16, :]
        output["current_iqr"] = np.quantile(current, 0.75, axis=1) - np.quantile(current, 0.25, axis=1)
        output["current_p90_p10"] = np.quantile(current, 0.90, axis=1) - np.quantile(current, 0.10, axis=1)
        output["recent_mean_change"] = current.mean(axis=1) - recent_start.mean(axis=1)
        output["recent_pattern_change"] = np.std(current - recent_start, axis=1)
        return output


@dataclass
class PreparedSamples:
    sequences: np.ndarray
    elevation: np.ndarray
    metadata: pd.DataFrame
    target_xel: np.ndarray
    target_el: np.ndarray
    sensor_cols: list[str]
    skip_counts: dict[str, int]


def build_windows(
    raw: pd.DataFrame,
    sensors: list[str],
    target_rows: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    sensor_frame = raw[["Timestamp"] + sensors].dropna(subset=["Timestamp"]).copy()
    sensor_frame["GridTime"] = sensor_frame["Timestamp"].dt.floor(f"{STEP_SECONDS}s")
    grid = sensor_frame.groupby("GridTime", sort=True)[sensors].mean().astype("float32")
    grid = grid.ffill(limit=3).bfill(limit=1)
    step = pd.Timedelta(seconds=STEP_SECONDS)
    sequences: list[np.ndarray] = []
    records: list[dict] = []
    skip = {"too_early": 0, "missing_grid": 0, "nan": 0, "thermal_jump": 0}

    for row in target_rows.itertuples(index=False):
        end = pd.Timestamp(row.Timestamp).floor(f"{STEP_SECONDS}s")
        expected = pd.date_range(end=end, periods=seq_len, freq=step)
        if expected[0] < grid.index.min():
            skip["too_early"] += 1
            continue
        if not expected.isin(grid.index).all():
            skip["missing_grid"] += 1
            continue
        values = grid.loc[expected, sensors].to_numpy(dtype="float32")
        if not np.isfinite(values).all():
            skip["nan"] += 1
            continue
        if np.abs(np.diff(values, axis=0)).max() > MAX_SENSOR_RATE_C:
            skip["thermal_jump"] += 1
            continue
        sequences.append(values)
        records.append({
            "Timestamp": row.Timestamp,
            "Elevation": row.Elevation,
            "Azimuth": row.Azimuth,
            "TargetXEL": row.TargetXEL,
        })
    return np.asarray(sequences, dtype="float32"), pd.DataFrame(records), skip


def prepare_samples(raw_path: Path, xel_audit_path: Path, pointing_audit_path: Path) -> PreparedSamples:
    raw = normalize_sensor_columns(pd.read_csv(raw_path, low_memory=False))
    raw["Timestamp"] = parse_timestamp(raw["Timestamp"])
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    sensors = sensor_columns(raw)
    if len(sensors) != 295:
        raise ValueError(f"Expected 295 sensors, found {len(sensors)}")

    xel_audit = pd.read_csv(xel_audit_path, low_memory=False)
    xel_audit["Timestamp"] = parse_timestamp(xel_audit["Timestamp"])
    xel_audit = xel_audit.dropna(subset=["Timestamp", "projected_true_XEL"])
    xel_audit = xel_audit.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    xel_index = xel_audit.set_index("Timestamp")

    target_mask = raw["XEl Offset"].notna()
    raw.loc[target_mask, "TargetXEL"] = raw.loc[target_mask, "Timestamp"].map(xel_index["projected_true_XEL"])
    raw.loc[target_mask, "ActiveAzelOffsetCount"] = raw.loc[target_mask, "Timestamp"].map(
        xel_index["azel_nonzero_count_since_zero"]
    )
    if raw.loc[target_mask, "TargetXEL"].isna().any():
        raise ValueError("Some pointing targets do not have a corrected XEL audit value")
    usable = target_mask & (pd.to_numeric(raw["ActiveAzelOffsetCount"], errors="coerce").fillna(0) == 0)
    target_rows = raw.loc[usable, ["Timestamp", "SRT Elevation Ange[DEG]", "Azimuth", "TargetXEL"]].copy()
    target_rows = target_rows.rename(columns={"SRT Elevation Ange[DEG]": "Elevation"})

    sequences, metadata, skip = build_windows(raw, sensors, target_rows)
    metadata = metadata.merge(
        xel_audit[[
            "Timestamp", "receiver_class", "receiver_band", "fit_xel_sigma_arcsec",
            "fit_el_sigma_arcsec", "lonfit_hpbw_arcsec", "latfit_hpbw_arcsec",
        ]],
        on="Timestamp", how="left", validate="one_to_one",
    ).rename(columns={
        "receiver_class": "ReceiverCode",
        "receiver_band": "ReceiverBand",
        "fit_xel_sigma_arcsec": "FitXelSigmaArcsec",
        "fit_el_sigma_arcsec": "FitElSigmaArcsec",
        "lonfit_hpbw_arcsec": "LonFitHPBWArcsec",
        "latfit_hpbw_arcsec": "LatFitHPBWArcsec",
    })

    pointing = pd.read_csv(pointing_audit_path, usecols=["Time", "Offset_EL_arcsec"], low_memory=False)
    pointing["Time"] = parse_timestamp(pointing["Time"])
    pointing = pointing.dropna(subset=["Time"]).sort_values("Time").drop_duplicates("Time", keep="last")
    el_by_time = pointing.set_index("Time")["Offset_EL_arcsec"]
    metadata["TargetEL"] = metadata["Timestamp"].map(el_by_time)
    if metadata["TargetEL"].isna().any():
        raise ValueError(f"Missing EL targets: {int(metadata['TargetEL'].isna().sum())}")

    order = np.argsort(metadata["Timestamp"].to_numpy())
    metadata = metadata.iloc[order].reset_index(drop=True)
    sequences = sequences[order]
    metadata["Split"] = chronological_split(len(metadata))
    return PreparedSamples(
        sequences=sequences,
        elevation=metadata["Elevation"].to_numpy(dtype="float32").reshape(-1, 1),
        metadata=metadata,
        target_xel=metadata["TargetXEL"].to_numpy(dtype="float64"),
        target_el=metadata["TargetEL"].to_numpy(dtype="float64"),
        sensor_cols=sensors,
        skip_counts=skip,
    )


def build_final_feature_dataset(samples: PreparedSamples) -> tuple[pd.DataFrame, dict]:
    train_mask = samples.metadata["Split"].eq("train").to_numpy()
    feature_builder = ThermalFeatureBuilder().fit(samples.sequences[train_mask])
    features = feature_builder.transform(samples.sequences, samples.elevation, samples.metadata)
    receivers = receiver_one_hot(samples.metadata)
    features = pd.concat([features, receivers], axis=1).add_prefix("feature__")
    metadata_columns = [
        "Timestamp", "Split", "Elevation", "Azimuth", "ReceiverCode", "ReceiverBand",
    ]
    output = samples.metadata[metadata_columns].copy()
    output["target_XEL_arcsec"] = samples.target_xel
    output["target_EL_arcsec"] = samples.target_el
    output = pd.concat([output, features], axis=1)
    artifacts = {
        "feature_builder": feature_builder,
        "sensor_columns": samples.sensor_cols,
        "receiver_levels": RECEIVER_LEVELS,
        "feature_columns": features.columns.tolist(),
        "sequence_length": SEQ_LEN,
        "step_seconds": STEP_SECONDS,
    }
    return output, artifacts


def save_prepared_dataset(
    dataset: pd.DataFrame,
    artifacts: dict,
    skip_counts: dict[str, int],
    output_csv: Path,
    artifact_path: Path,
    report_path: Path,
) -> dict:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv, index=False)
    joblib.dump(artifacts, artifact_path)
    report = {
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "split_counts": dataset["Split"].value_counts().to_dict(),
        "receiver_counts": dataset["ReceiverBand"].value_counts(dropna=False).to_dict(),
        "skip_counts": skip_counts,
        "timestamp_min": str(dataset["Timestamp"].min()),
        "timestamp_max": str(dataset["Timestamp"].max()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def transform_new_window(
    window: np.ndarray,
    timestamp,
    elevation: float,
    azimuth: float,
    receiver_band: str,
    feature_artifacts: dict,
) -> pd.DataFrame:
    """Transform one validated 60x295 thermal window into the production feature row."""
    values = np.asarray(window, dtype="float32")
    expected_shape = (feature_artifacts["sequence_length"], len(feature_artifacts["sensor_columns"]))
    if values.shape != expected_shape:
        raise ValueError(f"Expected thermal window {expected_shape}, received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("The thermal window contains NaN or infinite values")
    if np.abs(np.diff(values, axis=0)).max() > MAX_SENSOR_RATE_C:
        raise ValueError("The thermal window contains an implausible sensor jump")
    meta = pd.DataFrame([{
        "Timestamp": pd.Timestamp(timestamp),
        "Elevation": float(elevation),
        "Azimuth": float(azimuth),
        "ReceiverBand": receiver_band if receiver_band in RECEIVER_LEVELS else "unknown",
    }])
    builder = feature_artifacts["feature_builder"]
    features = builder.transform(values[None, :, :], np.asarray([[elevation]], dtype="float32"), meta)
    features = pd.concat([features, receiver_one_hot(meta)], axis=1).add_prefix("feature__")
    return features.reindex(columns=feature_artifacts["feature_columns"])
