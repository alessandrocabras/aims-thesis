"""Cross-check pointing results against control and calibration logs."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median


LOG_TS_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<doy>\d{3})-(?P<hour>\d{2}):(?P<minute>\d{2}):"
    r"(?P<second>\d{2})(?:\.(?P<fraction>\d+))?"
)
DAT_TS_RE = re.compile(
    r"^(?P<year>\d{4})\.(?P<doy>\d{3})\.(?P<hour>\d{2}):(?P<minute>\d{2}):"
    r"(?P<second>\d{2})(?:\.(?P<fraction>\d+))?"
)
COMMAND_RE = re.compile(r"Command issued \{([^{}]+)\}")
FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def parse_doy_timestamp(match: re.Match[str] | None) -> datetime | None:
    if match is None:
        return None
    base = datetime.strptime(f"{match.group('year')}-{match.group('doy')}", "%Y-%j")
    fraction = (match.group("fraction") or "")[:6].ljust(6, "0")
    return base.replace(
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=int(match.group("second")),
        microsecond=int(fraction or 0),
    )


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def normalized_timestamp_key(value: str) -> str:
    text = value.strip()
    if re.match(r"^\d{8} ", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}{text[8:]}"
    return iso(parse_iso_timestamp(text))


def iso(value: datetime | None) -> str:
    return value.isoformat(sep=" ", timespec="milliseconds") if value else ""


def as_float(value: str) -> float | None:
    try:
        result = float(value.rstrip("dD"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return math.nan
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def interval_distance(t: datetime, start: datetime, end: datetime) -> float:
    if start <= t <= end:
        return 0.0
    return min(abs((t - start).total_seconds()), abs((t - end).total_seconds()))


@dataclass
class CommandEvent:
    timestamp: datetime
    key: str
    value: str
    log_path: str


@dataclass
class LogRecord:
    path: str
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    schedules: list[str] = field(default_factory=list)
    receiver_setups: list[str] = field(default_factory=list)
    command_count: int = 0


def parse_command(payload: str) -> tuple[str, str]:
    if "=" in payload:
        key, value = payload.split("=", 1)
        return key.strip(), value.strip()
    return payload.strip(), ""


def parse_logs(log_root: Path):
    log_records: list[LogRecord] = []
    command_events: list[CommandEvent] = []
    command_counts: Counter[str] = Counter()
    command_values: dict[str, Counter[str]] = defaultdict(Counter)

    for path in log_root.rglob("*.log"):
        record = LogRecord(path=str(path))
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                timestamp = parse_doy_timestamp(LOG_TS_RE.match(line))
                if timestamp is not None:
                    if record.first_timestamp is None:
                        record.first_timestamp = timestamp
                    record.last_timestamp = timestamp
                if timestamp is None or "Command issued" not in line:
                    continue
                for payload in COMMAND_RE.findall(line):
                    key, value = parse_command(payload)
                    event = CommandEvent(timestamp, key, value, str(path))
                    command_events.append(event)
                    record.command_count += 1
                    command_counts[key] += 1
                    command_values[key][value] += 1
                    if key == "startSchedule":
                        record.schedules.append(value)
                    elif key == "receiversSetup":
                        record.receiver_setups.append(value)
        if record.first_timestamp and record.last_timestamp:
            log_records.append(record)

    command_events.sort(key=lambda event: event.timestamp)
    log_records.sort(key=lambda record: record.first_timestamp or datetime.max)
    return log_records, command_events, command_counts, command_values


def offset_values(value: str) -> tuple[float | None, float | None]:
    parts = value.split(",")
    if len(parts) < 2:
        return None, None
    return as_float(parts[0].strip()), as_float(parts[1].strip())


def build_reset_events(command_events: list[CommandEvent]):
    offset_keys = {"azelOffsets", "radecOffsets", "lonlatOffsets"}
    offset_events = []
    receiver_events = []
    for event in command_events:
        if event.key == "receiversSetup":
            receiver_events.append(event)
        if event.key not in offset_keys:
            continue
        first, second = offset_values(event.value)
        is_zero = (
            first is not None
            and second is not None
            and abs(first) < 1e-12
            and abs(second) < 1e-12
        )
        offset_events.append((event, is_zero, first, second))

    zero_events = [item for item in offset_events if item[1]]
    reset_events: list[datetime] = []
    start = 0
    for idx, (event, _, _, _) in enumerate(zero_events):
        while (event.timestamp - zero_events[start][0].timestamp).total_seconds() > 8:
            start += 1
        keys = {zero_events[j][0].key for j in range(start, idx + 1)}
        if keys == offset_keys:
            candidate = event.timestamp
            if not reset_events or (candidate - reset_events[-1]).total_seconds() > 8:
                reset_events.append(candidate)
    zero_times_by_key = {
        key: sorted(item[0].timestamp for item in offset_events if item[0].key == key and item[1])
        for key in offset_keys
    }
    nonzero_times_by_key = {
        key: sorted(item[0].timestamp for item in offset_events if item[0].key == key and not item[1])
        for key in offset_keys
    }
    return (
        offset_events,
        receiver_events,
        reset_events,
        zero_times_by_key,
        nonzero_times_by_key,
    )


def parse_dat_files(dat_root: Path):
    rows = []
    by_basename: dict[str, list[dict]] = defaultdict(list)
    for path in dat_root.rglob("*.dat"):
        row = {
            "dat_path": str(path),
            "dat_filename": path.name,
            "dat_first_timestamp": None,
            "dat_last_timestamp": None,
            "dat_project": "",
            "dat_observer": "",
            "dat_source": "",
            "dat_xoffset_timestamp": None,
            "dat_azimuth": None,
            "dat_elevation": None,
            "dat_xel_arcsec": None,
            "dat_el_arcsec": None,
            "dat_lon_offset_arcsec": None,
            "dat_lat_offset_arcsec": None,
            "dat_xel_sigma_arcsec": None,
            "dat_el_sigma_arcsec": None,
            "dat_fit_flag_1": "",
            "dat_fit_flag_2": "",
            "dat_latfit_raw": "",
            "dat_lonfit_raw": "",
            "dat_latfit_hpbw_arcsec": None,
            "dat_lonfit_hpbw_arcsec": None,
        }
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                timestamp = parse_doy_timestamp(DAT_TS_RE.match(line))
                if timestamp is not None:
                    if row["dat_first_timestamp"] is None:
                        row["dat_first_timestamp"] = timestamp
                    row["dat_last_timestamp"] = timestamp
                stripped = line.strip()
                if "#Project Name:" in stripped:
                    row["dat_project"] = stripped.split("#Project Name:", 1)[1].strip()
                elif "#Observer Name:" in stripped:
                    row["dat_observer"] = stripped.split("#Observer Name:", 1)[1].strip()
                elif "#fivpt#source " in stripped:
                    row["dat_source"] = stripped.split("#fivpt#source ", 1)[1].strip()
                elif "#fivpt#latfit " in stripped:
                    fit_payload = stripped.split("#fivpt#latfit ", 1)[1].strip()
                    row["dat_latfit_raw"] = fit_payload
                    fit_fields = fit_payload.split()
                    if len(fit_fields) >= 2:
                        hpbw = as_float(fit_fields[1])
                        row["dat_latfit_hpbw_arcsec"] = hpbw * 3600 if hpbw is not None else None
                elif "#fivpt#lonfit " in stripped:
                    fit_payload = stripped.split("#fivpt#lonfit ", 1)[1].strip()
                    row["dat_lonfit_raw"] = fit_payload
                    fit_fields = fit_payload.split()
                    if len(fit_fields) >= 2:
                        hpbw = as_float(fit_fields[1])
                        row["dat_lonfit_hpbw_arcsec"] = hpbw * 3600 if hpbw is not None else None
                elif "#fivpt#xoffset " in stripped:
                    payload = stripped.split("#fivpt#xoffset ", 1)[1].split()
                    if len(payload) >= 4:
                        row["dat_xoffset_timestamp"] = timestamp
                        row["dat_azimuth"] = as_float(payload[0])
                        row["dat_elevation"] = as_float(payload[1])
                        xel_deg = as_float(payload[2])
                        el_deg = as_float(payload[3])
                        row["dat_xel_arcsec"] = xel_deg * 3600 if xel_deg is not None else None
                        row["dat_el_arcsec"] = el_deg * 3600 if el_deg is not None else None
                        if len(payload) >= 6:
                            xel_sigma = as_float(payload[4])
                            el_sigma = as_float(payload[5])
                            row["dat_xel_sigma_arcsec"] = (
                                xel_sigma * 3600 if xel_sigma is not None else None
                            )
                            row["dat_el_sigma_arcsec"] = (
                                el_sigma * 3600 if el_sigma is not None else None
                            )
                        if len(payload) >= 8:
                            row["dat_fit_flag_1"] = payload[-2]
                            row["dat_fit_flag_2"] = payload[-1]
                elif "#fivpt#offset " in stripped:
                    payload = stripped.split("#fivpt#offset ", 1)[1].split()
                    if len(payload) >= 4:
                        lon_deg = as_float(payload[2])
                        lat_deg = as_float(payload[3])
                        row["dat_lon_offset_arcsec"] = (
                            lon_deg * 3600 if lon_deg is not None else None
                        )
                        row["dat_lat_offset_arcsec"] = (
                            lat_deg * 3600 if lat_deg is not None else None
                        )
        rows.append(row)
        by_basename[path.name.lower()].append(row)
    return rows, by_basename


def find_matching_log(t: datetime, log_records: list[LogRecord]) -> tuple[LogRecord | None, float]:
    best = None
    best_distance = math.inf
    for record in log_records:
        assert record.first_timestamp is not None and record.last_timestamp is not None
        distance = interval_distance(t, record.first_timestamp, record.last_timestamp)
        if distance < best_distance:
            best = record
            best_distance = distance
    return best, best_distance


def previous_event(events: list, times: list[datetime], t: datetime):
    idx = bisect.bisect_right(times, t) - 1
    return events[idx] if idx >= 0 else None


def any_event_between(times: list[datetime], start: datetime | None, end: datetime) -> bool:
    if start is None:
        return False
    idx = bisect.bisect_right(times, start)
    return idx < len(times) and times[idx] <= end


def classify_receiver(raw_receiver: str, context: str) -> str:
    if raw_receiver:
        return raw_receiver
    text = f"{raw_receiver} {context}".upper()
    if any(token in text for token in ("K18", "K25", "KKG", "KBAND", "K-BAND")):
        return "K"
    if any(token in text for token in ("CLOW", "C-LOW", "CLOWBAND")):
        return "C_low"
    if any(token in text for token in ("CHIGH", "C-HIGH", "CHIGHBAND")):
        return "C_high"
    if "CCB" in text or "CBAND" in text or "C-BAND" in text:
        return "C_unspecified"
    return "unknown"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows and not fieldnames:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows: list[dict], field: str):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, ""))].append(row)
    result = []
    for key, subset in sorted(groups.items()):
        abs_xel = [abs(float(r["Offset_XEL_arcsec"])) for r in subset]
        abs_el = [abs(float(r["Offset_EL_arcsec"])) for r in subset]
        result.append(
            {
                "group_field": field,
                "group_value": key,
                "count": len(subset),
                "median_abs_xel": median(abs_xel),
                "p90_abs_xel": quantile(abs_xel, 0.9),
                "median_abs_el": median(abs_el),
                "p90_abs_el": quantile(abs_el, 0.9),
                "gross_invalid_percent": 100 * sum(bool(r["gross_invalid"]) for r in subset) / len(subset),
            }
        )
    return result


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointing", type=Path, default=project_root / "data" / "raw" / "control" / "pointing_offsets_real_azimuth.csv")
    parser.add_argument("--logs", type=Path, default=project_root / "data" / "raw" / "control" / "logs")
    parser.add_argument("--dat", type=Path, default=project_root / "data" / "raw" / "control" / "extradata")
    parser.add_argument(
        "--effective",
        type=Path,
        default=project_root / "data" / "raw" / "real" / "EffectiveDataset_Tseries_with_azimuth.csv",
    )
    parser.add_argument("--output", type=Path, default=project_root / "data" / "diagnostics" / "control" / "control_crosscheck")
    args = parser.parse_args()

    print(f"Parsing logs: {args.logs.resolve()}")
    log_records, command_events, command_counts, command_values = parse_logs(args.logs)
    (
        offset_events,
        receiver_events,
        reset_events,
        zero_times_by_key,
        nonzero_times_by_key,
    ) = build_reset_events(command_events)
    print(
        f"Valid logs={len(log_records)} | commands={len(command_events)} | "
        f"three-command resets={len(reset_events)} | receiver setup={len(receiver_events)}"
    )

    print(f"Parsing calibration data: {args.dat.resolve()}")
    dat_rows, dat_by_basename = parse_dat_files(args.dat)
    print(f"DAT={len(dat_rows)} | fivpt xoffset={sum(r['dat_xoffset_timestamp'] is not None for r in dat_rows)}")

    command_inventory = []
    for key, count in command_counts.most_common():
        for value, value_count in command_values[key].most_common(20):
            command_inventory.append(
                {"command": key, "total_count": count, "value": value, "value_count": value_count}
            )
    write_csv(args.output.with_name(args.output.name + "_command_inventory.csv"), command_inventory)

    dat_export = []
    for row in dat_rows:
        exported = dict(row)
        for key in ("dat_first_timestamp", "dat_last_timestamp", "dat_xoffset_timestamp"):
            exported[key] = iso(exported[key])
        dat_export.append(exported)
    write_csv(args.output.with_name(args.output.name + "_dat_audit.csv"), dat_export)

    with args.pointing.open("r", newline="", encoding="utf-8-sig") as handle:
        pointing_rows = list(csv.DictReader(handle))
    pointing_rows.sort(key=lambda row: parse_iso_timestamp(row["Time"]))

    receiver_times = [event.timestamp for event in receiver_events]
    receiver_events_by_log: dict[str, list[CommandEvent]] = defaultdict(list)
    for event in receiver_events:
        receiver_events_by_log[event.log_path].append(event)
    reset_times = reset_events
    nonzero_offset_events = [item for item in offset_events if not item[1]]
    nonzero_times = [item[0].timestamp for item in nonzero_offset_events]
    output_rows = []
    previous_scan_time = None
    azel_events = [item for item in offset_events if item[0].key == "azelOffsets"]
    azel_pointer = 0
    azel_latest_az = 0.0
    azel_latest_el = 0.0
    azel_cumulative_az = 0.0
    azel_cumulative_el = 0.0
    azel_nonzero_count_since_zero = 0
    azel_last_command_time = None

    for row in pointing_rows:
        t = parse_iso_timestamp(row["Time"])
        while azel_pointer < len(azel_events) and azel_events[azel_pointer][0].timestamp <= t:
            event, is_zero, first, second = azel_events[azel_pointer]
            if first is not None and second is not None:
                azel_latest_az, azel_latest_el = first, second
                if is_zero:
                    azel_cumulative_az = 0.0
                    azel_cumulative_el = 0.0
                    azel_nonzero_count_since_zero = 0
                else:
                    azel_cumulative_az += first
                    azel_cumulative_el += second
                    azel_nonzero_count_since_zero += 1
                azel_last_command_time = event.timestamp
            azel_pointer += 1
        dat_matches = dat_by_basename.get(row["Filename"].lower(), [])
        dat_row = min(
            dat_matches,
            key=lambda item: abs(((item["dat_xoffset_timestamp"] or item["dat_last_timestamp"] or t) - t).total_seconds()),
        ) if dat_matches else None
        log_record, log_distance = find_matching_log(t, log_records)
        log_covered = log_record is not None and log_distance <= 15 * 60

        local_receivers = receiver_events_by_log.get(log_record.path, []) if log_record else []
        local_receiver_times = [event.timestamp for event in local_receivers]
        previous_receiver = previous_event(local_receivers, local_receiver_times, t)
        receiver_age_h = (
            (t - previous_receiver.timestamp).total_seconds() / 3600 if previous_receiver else math.nan
        )
        raw_receiver = (
            previous_receiver.value if previous_receiver and receiver_age_h <= 24 else ""
        )
        context = " ".join(
            [
                row["Filename"],
                str(dat_row.get("dat_path", "")) if dat_row else "",
                str(dat_row.get("dat_project", "")) if dat_row else "",
                " ".join(log_record.schedules) if log_record else "",
                Path(log_record.path).name if log_record else "",
            ]
        )
        receiver_class = classify_receiver(raw_receiver, context)

        last_reset = previous_event(reset_times, reset_times, t)
        minutes_since_reset = (
            (t - last_reset).total_seconds() / 60 if last_reset is not None else math.nan
        )
        reset_between_scans = any_event_between(reset_times, previous_scan_time, t)
        nonzero_after_reset = (
            last_reset is not None and any_event_between(nonzero_times, last_reset, t)
        )
        last_nonzero = previous_event(nonzero_offset_events, nonzero_times, t)
        minutes_since_nonzero = (
            (t - last_nonzero[0].timestamp).total_seconds() / 60 if last_nonzero else math.nan
        )

        xel = float(row["Offset_XEL_arcsec"])
        el = float(row["Offset_EL_arcsec"])
        gross_invalid = abs(xel) > 60 or abs(el) > 180
        result = dict(row)
        result.update(
            {
                "dat_found": bool(dat_row),
                "dat_duplicate_count": len(dat_matches),
                "dat_time_delta_seconds": (
                    abs(((dat_row["dat_xoffset_timestamp"] or dat_row["dat_last_timestamp"]) - t).total_seconds())
                    if dat_row and (dat_row["dat_xoffset_timestamp"] or dat_row["dat_last_timestamp"])
                    else ""
                ),
                "dat_xel_delta_arcsec": (
                    dat_row["dat_xel_arcsec"] - xel
                    if dat_row and dat_row["dat_xel_arcsec"] is not None
                    else ""
                ),
                "dat_lon_delta_arcsec": (
                    dat_row["dat_lon_offset_arcsec"] - xel
                    if dat_row and dat_row["dat_lon_offset_arcsec"] is not None
                    else ""
                ),
                "dat_el_delta_arcsec": (
                    dat_row["dat_el_arcsec"] - el
                    if dat_row and dat_row["dat_el_arcsec"] is not None
                    else ""
                ),
                "dat_xel_sigma_arcsec": dat_row["dat_xel_sigma_arcsec"] if dat_row else "",
                "dat_el_sigma_arcsec": dat_row["dat_el_sigma_arcsec"] if dat_row else "",
                "dat_fit_flag_1": dat_row["dat_fit_flag_1"] if dat_row else "",
                "dat_fit_flag_2": dat_row["dat_fit_flag_2"] if dat_row else "",
                "dat_latfit_hpbw_arcsec": dat_row["dat_latfit_hpbw_arcsec"] if dat_row else "",
                "dat_lonfit_hpbw_arcsec": dat_row["dat_lonfit_hpbw_arcsec"] if dat_row else "",
                "dat_project": dat_row["dat_project"] if dat_row else "",
                "dat_observer": dat_row["dat_observer"] if dat_row else "",
                "dat_source": dat_row["dat_source"] if dat_row else "",
                "log_covered_15min": log_covered,
                "matched_log": Path(log_record.path).name if log_record else "",
                "log_interval_distance_seconds": log_distance if log_record else "",
                "log_schedules": " | ".join(log_record.schedules) if log_record else "",
                "receiver_setup_raw": raw_receiver,
                "receiver_setup_age_hours": receiver_age_h,
                "receiver_class": receiver_class,
                "minutes_since_complete_offset_reset": minutes_since_reset,
                "complete_reset_between_previous_scan": reset_between_scans,
                "azel_zero_between_previous_scan": any_event_between(
                    zero_times_by_key["azelOffsets"], previous_scan_time, t
                ),
                "azel_nonzero_between_previous_scan": any_event_between(
                    nonzero_times_by_key["azelOffsets"], previous_scan_time, t
                ),
                "nonzero_offset_after_last_reset": nonzero_after_reset,
                "minutes_since_nonzero_offset_command": minutes_since_nonzero,
                "azel_latest_az_deg": azel_latest_az,
                "azel_latest_el_deg": azel_latest_el,
                "azel_cumulative_az_since_zero_deg": azel_cumulative_az,
                "azel_cumulative_el_since_zero_deg": azel_cumulative_el,
                "azel_nonzero_count_since_zero": azel_nonzero_count_since_zero,
                "minutes_since_last_azel_command": (
                    (t - azel_last_command_time).total_seconds() / 60
                    if azel_last_command_time is not None
                    else ""
                ),
                "lon_plus_latest_azel_arcsec": xel + 3600 * azel_latest_az,
                "lon_plus_cumulative_azel_arcsec": xel + 3600 * azel_cumulative_az,
                "el_plus_latest_azel_arcsec": el + 3600 * azel_latest_el,
                "el_plus_cumulative_azel_arcsec": el + 3600 * azel_cumulative_el,
                "previous_scan_gap_minutes": (
                    (t - previous_scan_time).total_seconds() / 60 if previous_scan_time else ""
                ),
                "gross_invalid": gross_invalid,
            }
        )
        output_rows.append(result)
        previous_scan_time = t

    output_path = args.output.with_name(args.output.name + "_pointing_audit.csv")
    write_csv(output_path, output_rows)

    effective_audit = []
    if args.effective.exists():
        audit_by_time = {normalized_timestamp_key(row["Time"]): row for row in output_rows}
        with args.effective.open("r", newline="", encoding="utf-8-sig") as handle:
            for source in csv.DictReader(handle):
                stored = as_float(source.get("XEl Offset", ""))
                if stored is None:
                    continue
                elevation = as_float(source.get("SRT Elevation Ange[DEG]", ""))
                timestamp = source.get("Timestamp", "")
                matched_row = audit_by_time.get(normalized_timestamp_key(timestamp))
                projected = (
                    stored * math.cos(math.radians(elevation)) if elevation is not None else math.nan
                )
                effective_audit.append(
                    {
                        "Timestamp": timestamp,
                        "Elevation": elevation,
                        "Azimuth": source.get("Azimuth", ""),
                        "stored_XEl_Offset": stored,
                        "projected_true_XEL": projected,
                        "master_matched": bool(matched_row),
                        "stored_minus_master_lon": (
                            stored - float(matched_row["Offset_XEL_arcsec"])
                            if matched_row
                            else ""
                        ),
                        "projected_minus_dat_xoffset": (
                            projected
                            - (
                                float(matched_row["Offset_XEL_arcsec"])
                                + float(matched_row["dat_xel_delta_arcsec"])
                            )
                            if matched_row and matched_row["dat_xel_delta_arcsec"] != ""
                            else ""
                        ),
                        "receiver_class": matched_row["receiver_class"] if matched_row else "",
                        "receiver_band": (
                            {
                                "KKG": "K",
                                "K": "K",
                                "CCB": "C_high",
                                "CCG": "C_low",
                                "QQG": "Q",
                            }.get(matched_row["receiver_class"], "unknown")
                            if matched_row
                            else ""
                        ),
                        "azel_nonzero_count_since_zero": (
                            matched_row["azel_nonzero_count_since_zero"] if matched_row else ""
                        ),
                        "azel_latest_az_deg": matched_row["azel_latest_az_deg"] if matched_row else "",
                        "azel_latest_el_deg": matched_row["azel_latest_el_deg"] if matched_row else "",
                        "fit_xel_sigma_arcsec": matched_row["dat_xel_sigma_arcsec"] if matched_row else "",
                        "fit_el_sigma_arcsec": matched_row["dat_el_sigma_arcsec"] if matched_row else "",
                        "lonfit_hpbw_arcsec": matched_row["dat_lonfit_hpbw_arcsec"] if matched_row else "",
                        "latfit_hpbw_arcsec": matched_row["dat_latfit_hpbw_arcsec"] if matched_row else "",
                        "fit_flag_1": matched_row["dat_fit_flag_1"] if matched_row else "",
                        "fit_flag_2": matched_row["dat_fit_flag_2"] if matched_row else "",
                        "gross_invalid": matched_row["gross_invalid"] if matched_row else "",
                    }
                )
        write_csv(
            args.output.with_name(args.output.name + "_effective_target_projection.csv"),
            effective_audit,
        )

    summary = []
    for field in (
        "dat_found",
        "log_covered_15min",
        "receiver_class",
        "dat_fit_flag_1",
        "dat_fit_flag_2",
        "complete_reset_between_previous_scan",
        "azel_zero_between_previous_scan",
        "azel_nonzero_between_previous_scan",
        "nonzero_offset_after_last_reset",
        "gross_invalid",
    ):
        summary.extend(group_summary(output_rows, field))
    write_csv(args.output.with_name(args.output.name + "_summary.csv"), summary)

    receiver_setup_counts = Counter(event.value for event in receiver_events)
    matched = [r for r in output_rows if r["dat_found"]]
    covered = [r for r in output_rows if r["log_covered_15min"]]
    reset_known = [r for r in output_rows if r["complete_reset_between_previous_scan"]]
    no_reset_near = [
        r for r in output_rows
        if r["previous_scan_gap_minutes"] != ""
        and float(r["previous_scan_gap_minutes"]) <= 180
        and not r["complete_reset_between_previous_scan"]
    ]
    print('\n=== COVERAGE ===')
    print(f"Pointing master: {len(output_rows)}")
    print(f"DAT matched: {len(matched)} ({100*len(matched)/len(output_rows):.1f}%)")
    print(f"Logs within 15 min: {len(covered)} ({100*len(covered)/len(output_rows):.1f}%)")
    print(f"Complete resets between consecutive scans: {len(reset_known)}")
    print(f"Scans within 3h without an observed complete reset: {len(no_reset_near)}")
    print("Receiver setup:", dict(receiver_setup_counts.most_common()))
    print("Receiver class master:", dict(Counter(r["receiver_class"] for r in output_rows)))
    print("Fit flag 1:", dict(Counter(r["dat_fit_flag_1"] for r in matched)))
    print("Fit flag 2:", dict(Counter(r["dat_fit_flag_2"] for r in matched)))

    if matched:
        max_dx = max(abs(float(r["dat_xel_delta_arcsec"])) for r in matched if r["dat_xel_delta_arcsec"] != "")
        max_dlon = max(abs(float(r["dat_lon_delta_arcsec"])) for r in matched if r["dat_lon_delta_arcsec"] != "")
        max_de = max(abs(float(r["dat_el_delta_arcsec"])) for r in matched if r["dat_el_delta_arcsec"] != "")
        print(
            f"DAT/master agreement: max |dXEL|={max_dx:.6g}, "
            f"max |dLon|={max_dlon:.6g}, max |dEL|={max_de:.6g}"
        )

    if effective_audit:
        stored_abs = [abs(float(row["stored_XEl_Offset"])) for row in effective_audit]
        projected_abs = [abs(float(row["projected_true_XEL"])) for row in effective_audit]
        master_deltas = [
            abs(float(row["stored_minus_master_lon"]))
            for row in effective_audit
            if row["stored_minus_master_lon"] != ""
        ]
        projection_deltas = [
            abs(float(row["projected_minus_dat_xoffset"]))
            for row in effective_audit
            if row["projected_minus_dat_xoffset"] != ""
        ]
        max_master_delta = max(master_deltas) if master_deltas else math.nan
        max_projection_delta = max(projection_deltas) if projection_deltas else math.nan
        print('\n=== EFFECTIVE-DATASET TARGETS ===')
        print(f"Valid targets: {len(effective_audit)}")
        print(
            f"Stored (lon): median |y|={median(stored_abs):.3f}, "
            f"p90={quantile(stored_abs, 0.9):.3f}, |y|>15={100*sum(v>15 for v in stored_abs)/len(stored_abs):.1f}%"
        )
        print(
            f"Projected XEL: median |y|={median(projected_abs):.3f}, "
            f"p90={quantile(projected_abs, 0.9):.3f}, |y|>15={100*sum(v>15 for v in projected_abs)/len(projected_abs):.1f}%"
        )
        print(
            f"max |stored-master lon|={max_master_delta:.6g}; "
            f"max |projected-DAT xoffset|={max_projection_delta:.6g}"
        )

    print("\nOutput:")
    print(f"- {output_path}")
    print(f"- {args.output.with_name(args.output.name + '_summary.csv')}")
    print(f"- {args.output.with_name(args.output.name + '_command_inventory.csv')}")
    print(f"- {args.output.with_name(args.output.name + '_dat_audit.csv')}")
    if effective_audit:
        print(f"- {args.output.with_name(args.output.name + '_effective_target_projection.csv')}")


if __name__ == "__main__":
    main()
