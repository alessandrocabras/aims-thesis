"""Utilities for an auditable FEM-v1/FEM-v2 pretraining comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_TOLERANCE = pd.Timedelta(milliseconds=2)


def attach_fem_v2_labels(frame: pd.DataFrame, labels_path: str | Path) -> pd.DataFrame:
    """Attach v2 labels before any row filtering and return the matched subset.

    The source row ID is the one-based Excel row number, including the header.
    Both FEM variants must use the returned subset for a fair comparison.
    """
    required = {"Time", "Elevation", "Accuracy_1", "Accuracy_2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing FEM source columns: {sorted(missing)}")
    source = frame.copy()
    if "source_row_id" not in source:
        source.insert(0, "source_row_id", np.arange(2, len(source) + 2, dtype=np.int64))
    labels = pd.read_csv(labels_path)
    if not labels["source_row_id"].is_unique:
        raise ValueError("FEM-v2 source_row_id is not unique")
    matched = source.merge(labels, on="source_row_id", how="inner", validate="one_to_one", suffixes=("", "_v2audit"))
    if len(matched) != len(labels):
        raise ValueError(f"FEM-v2 join lost rows: expected {len(labels)}, found {len(matched)}")
    elevation = pd.to_numeric(matched["Elevation"], errors="raise").to_numpy(float)
    audit_elevation = pd.to_numeric(matched["Elevation_v2audit"], errors="raise").to_numpy(float)
    if not np.array_equal(elevation, audit_elevation):
        raise ValueError("Elevation mismatch after FEM-v2 join")
    time = pd.to_datetime(matched["Time"], format="mixed", errors="raise")
    audit_time = pd.to_datetime(matched["Timestamp"], format="mixed", errors="raise")
    time_delta = (time - audit_time).abs()
    max_time_delta = time_delta.max()
    if max_time_delta > TIMESTAMP_TOLERANCE:
        mismatch = matched.loc[time_delta > TIMESTAMP_TOLERANCE, [
            "source_row_id", "Time", "Timestamp",
        ]].head(5).copy()
        mismatch["abs_delta_ms"] = (
            time_delta.loc[mismatch.index] / pd.Timedelta(milliseconds=1)
        ).to_numpy()
        raise ValueError(
            "Timestamp mismatch after FEM-v2 join: "
            f"max absolute delta={max_time_delta}; examples={mismatch.to_dict(orient='records')}"
        )
    matched.attrs["timestamp_alignment_max_abs_ms"] = float(
        max_time_delta / pd.Timedelta(milliseconds=1)
    )
    expected_v1_xel = pd.to_numeric(matched["Accuracy_1"], errors="raise").to_numpy(float) * np.cos(
        np.deg2rad(elevation)
    )
    expected_v1_el = pd.to_numeric(matched["Accuracy_2"], errors="raise").to_numpy(float)
    if not np.allclose(expected_v1_xel, matched["FEMv1_XEL_arcsec"], rtol=0, atol=1e-8):
        raise ValueError("Historical XEL audit mismatch; do not use the v2 join")
    if not np.allclose(expected_v1_el, matched["FEMv1_EL_arcsec"], rtol=0, atol=1e-8):
        raise ValueError("Historical EL audit mismatch; do not use the v2 join")
    return matched


def target_columns(variant: str) -> tuple[str, str]:
    """Return XEL and EL target columns for a matched frame."""
    variants = {
        "v1": ("FEMv1_XEL_arcsec", "FEMv1_EL_arcsec"),
        "v2": ("FEMv2_XEL_arcsec", "FEMv2_EL_arcsec"),
    }
    try:
        return variants[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown FEM label variant: {variant!r}") from exc
