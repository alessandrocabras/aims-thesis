"""Summarize complete DAT cross-scans excluded from the original pointing master."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

from audit_control_crossscan import offset_values, parse_logs


ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "data" / "raw" / "control"
DIAGNOSTICS = ROOT / "data" / "diagnostics" / "control"
FULL_AUDIT = DIAGNOSTICS / "all_completed_dat_crosscheck_pointing_audit.csv"
MASTER = CONTROL / "pointing_offsets_real_azimuth.csv"


def receiver_band(receiver: object) -> str:
    value = str(receiver).upper()
    if value in {"KKG", "K", "KQW", "KWQ", "KKQ"}:
        return "K"
    if value == "CCB":
        return "C_high"
    if value == "CCG":
        return "C_low"
    if value == "QQG":
        return "Q"
    return "unknown"


def robust_upper(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    median = clean.median()
    mad = (clean - median).abs().median()
    return float(median + 6 * max(mad, 1e-6))


def main() -> None:
    audit = pd.read_csv(FULL_AUDIT, low_memory=False)
    master = pd.read_csv(MASTER, usecols=["Filename"])
    master_names = set(master["Filename"].str.lower())

    audit["Timestamp"] = pd.to_datetime(audit["Time"], errors="coerce")
    audit["is_master"] = audit["Filename"].str.lower().isin(master_names)
    audit["ReceiverBand"] = audit["receiver_class"].map(receiver_band)
    audit["source_name"] = audit["dat_source"].fillna("").str.split().str[0].str.lower()
    audit = audit.sort_values("Timestamp").reset_index(drop=True)

    for direction, shift in (("previous", 1), ("next", -1)):
        adjacent_time = audit["Timestamp"].shift(shift)
        audit[f"{direction}_gap_minutes"] = (
            (audit["Timestamp"] - adjacent_time).abs().dt.total_seconds() / 60
        )
        audit[f"{direction}_is_master"] = (
            audit["is_master"].shift(shift).astype("boolean").fillna(False).astype(bool)
        )
        audit[f"{direction}_same_receiver"] = (
            audit["ReceiverBand"].eq(audit["ReceiverBand"].shift(shift))
        )
        audit[f"{direction}_same_source"] = (
            audit["source_name"].ne("")
            & audit["source_name"].eq(audit["source_name"].shift(shift))
        )
        audit[f"{direction}_filename"] = audit["Filename"].shift(shift)

    reference = audit[audit["is_master"]].copy()
    thresholds = {}
    for band, subset in reference.groupby("ReceiverBand"):
        thresholds[band] = {
            "lon_hpbw_median": subset["dat_lonfit_hpbw_arcsec"].median(),
            "lat_hpbw_median": subset["dat_latfit_hpbw_arcsec"].median(),
            "xel_sigma_high": robust_upper(subset["dat_xel_sigma_arcsec"]),
            "el_sigma_high": robust_upper(subset["dat_el_sigma_arcsec"]),
        }

    extra = audit[~audit["is_master"]].copy()
    _, command_events, _, _ = parse_logs(CONTROL / "logs")
    offset_keys = ("azelOffsets", "radecOffsets", "lonlatOffsets")
    for key in offset_keys:
        events = [event for event in command_events if event.key == key]
        event_times = [event.timestamp for event in events]
        first_values = []
        second_values = []
        active_values = []
        for timestamp in extra["Timestamp"]:
            index = bisect_right(event_times, timestamp.to_pydatetime()) - 1
            first, second = offset_values(events[index].value) if index >= 0 else (None, None)
            first_values.append(first)
            second_values.append(second)
            active_values.append(
                first is not None
                and second is not None
                and (abs(first) > 1e-12 or abs(second) > 1e-12)
            )
        stem = key.removesuffix("Offsets").lower()
        extra[f"{stem}_offset_1_deg"] = first_values
        extra[f"{stem}_offset_2_deg"] = second_values
        extra[f"active_{stem}_offset"] = active_values
    extra["active_any_commanded_offset"] = extra[
        ["active_azel_offset", "active_radec_offset", "active_lonlat_offset"]
    ].any(axis=1)
    extra["fit_flag_1_ok"] = pd.to_numeric(extra["dat_fit_flag_1"], errors="coerce").eq(1)
    extra["fit_flag_2_ok"] = pd.to_numeric(extra["dat_fit_flag_2"], errors="coerce").eq(1)
    extra["both_fit_flags_ok"] = extra["fit_flag_1_ok"] & extra["fit_flag_2_ok"]
    extra["procedural_followup_60m"] = (
        extra["previous_gap_minutes"].le(60)
        & extra["previous_same_receiver"]
        & extra["previous_same_source"]
    )
    extra["failed_attempt_before_master_retry_60m"] = (
        extra["next_gap_minutes"].le(60)
        & extra["next_is_master"]
        & extra["next_same_receiver"]
        & extra["next_same_source"]
    )

    def quality_flags(row: pd.Series) -> pd.Series:
        threshold = thresholds.get(row["ReceiverBand"])
        if not threshold:
            return pd.Series({"hpbw_anomalous": True, "sigma_anomalous": True})
        lon = row["dat_lonfit_hpbw_arcsec"]
        lat = row["dat_latfit_hpbw_arcsec"]
        hpbw_bad = (
            not np.isfinite(lon)
            or not np.isfinite(lat)
            or lon <= 0
            or lat <= 0
            or not 0.5 * threshold["lon_hpbw_median"] <= lon <= 1.5 * threshold["lon_hpbw_median"]
            or not 0.5 * threshold["lat_hpbw_median"] <= lat <= 1.5 * threshold["lat_hpbw_median"]
        )
        xel_sigma = row["dat_xel_sigma_arcsec"]
        el_sigma = row["dat_el_sigma_arcsec"]
        sigma_bad = (
            not np.isfinite(xel_sigma)
            or not np.isfinite(el_sigma)
            or xel_sigma < 0
            or el_sigma < 0
            or xel_sigma > threshold["xel_sigma_high"]
            or el_sigma > threshold["el_sigma_high"]
        )
        return pd.Series({"hpbw_anomalous": hpbw_bad, "sigma_anomalous": sigma_bad})

    extra[["hpbw_anomalous", "sigma_anomalous"]] = extra.apply(quality_flags, axis=1)

    def reasons(row: pd.Series) -> str:
        result = []
        if not row["fit_flag_1_ok"]:
            result.append("fit_flag_1_failed")
        if not row["fit_flag_2_ok"]:
            result.append("fit_flag_2_failed")
        if row["active_any_commanded_offset"]:
            result.append("active_commanded_offset")
        if row["gross_invalid"]:
            result.append("gross_offset")
        if row["hpbw_anomalous"]:
            result.append("hpbw_anomalous")
        if row["sigma_anomalous"]:
            result.append("sigma_anomalous")
        if row["procedural_followup_60m"]:
            result.append("procedural_followup")
        if row["failed_attempt_before_master_retry_60m"]:
            result.append("followed_by_accepted_retry")
        return "|".join(result)

    extra["exclusion_evidence"] = extra.apply(reasons, axis=1)

    export_columns = [
        "Filename", "Timestamp", "dat_project", "source_name", "Azimuth", "Elevation",
        "Offset_XEL_arcsec", "Offset_EL_arcsec", "receiver_class", "ReceiverBand",
        "receiver_setup_raw", "log_covered_15min", "active_azel_offset",
        "active_radec_offset", "active_lonlat_offset", "active_any_commanded_offset",
        "azel_offset_1_deg", "azel_offset_2_deg", "radec_offset_1_deg",
        "radec_offset_2_deg", "lonlat_offset_1_deg", "lonlat_offset_2_deg",
        "azel_nonzero_count_since_zero", "azel_latest_az_deg", "azel_latest_el_deg",
        "dat_fit_flag_1", "dat_fit_flag_2", "both_fit_flags_ok",
        "dat_xel_sigma_arcsec", "dat_el_sigma_arcsec", "dat_lonfit_hpbw_arcsec",
        "dat_latfit_hpbw_arcsec", "hpbw_anomalous", "sigma_anomalous", "gross_invalid",
        "previous_gap_minutes", "previous_filename", "procedural_followup_60m",
        "next_gap_minutes", "next_filename", "failed_attempt_before_master_retry_60m",
        "exclusion_evidence",
    ]
    extra[export_columns].to_csv(
        DIAGNOSTICS / "additional_dat_crossscan_audit.csv", index=False
    )

    summary = pd.DataFrame([
        {"metric": "additional_complete_crossscans", "value": len(extra)},
        {"metric": "log_covered_15min", "value": int(extra["log_covered_15min"].sum())},
        {"metric": "both_fit_flags_ok", "value": int(extra["both_fit_flags_ok"].sum())},
        {"metric": "active_azel_offset", "value": int(extra["active_azel_offset"].sum())},
        {"metric": "active_radec_offset", "value": int(extra["active_radec_offset"].sum())},
        {"metric": "active_lonlat_offset", "value": int(extra["active_lonlat_offset"].sum())},
        {
            "metric": "active_any_commanded_offset",
            "value": int(extra["active_any_commanded_offset"].sum()),
        },
        {"metric": "gross_offset", "value": int(extra["gross_invalid"].sum())},
        {"metric": "hpbw_anomalous", "value": int(extra["hpbw_anomalous"].sum())},
        {"metric": "sigma_anomalous", "value": int(extra["sigma_anomalous"].sum())},
        {"metric": "procedural_followup_60m", "value": int(extra["procedural_followup_60m"].sum())},
        {
            "metric": "failed_attempt_before_master_retry_60m",
            "value": int(extra["failed_attempt_before_master_retry_60m"].sum()),
        },
    ])
    summary.to_csv(DIAGNOSTICS / "additional_dat_crossscan_summary.csv", index=False)

    receiver_summary = extra.groupby(["ReceiverBand", "receiver_class"], dropna=False).agg(
        count=("Filename", "size"),
        active_offset=("active_any_commanded_offset", "sum"),
        gross_offset=("gross_invalid", "sum"),
        hpbw_anomalous=("hpbw_anomalous", "sum"),
        sigma_anomalous=("sigma_anomalous", "sum"),
    ).reset_index()
    receiver_summary.to_csv(
        DIAGNOSTICS / "additional_dat_crossscan_by_receiver.csv", index=False
    )

    print(summary.to_string(index=False))
    print("\nReceiver summary")
    print(receiver_summary.to_string(index=False))


if __name__ == "__main__":
    main()
