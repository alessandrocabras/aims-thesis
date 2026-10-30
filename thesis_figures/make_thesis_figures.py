"""Render 13 thesis figures from verified frozen snapshots; --refresh-sources replaces the snapshots."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np


HERE = Path(__file__).resolve().parent
BLUE, ORANGE, TEAL = "#0072B2", "#D55E00", "#00866B"
INK, GREY = "#303840", "#66717A"
PALE_BLUE, PALE_TEAL, PALE_ORANGE = "#E8F2F8", "#E7F3EE", "#FCEDDF"
COLORS = {"XEL": BLUE, "EL": ORANGE}
STEMS = [
    "01_target_coordinate_audit", "02_final_data_pipeline",
    "13_dataset_timeline", "14_temporal_evaluation_design",
    "10_fem_real_thermal_shift", "04_final_prediction_pipeline_parallel",
    "15_matched_transfer_protocol", "16_fem_transfer_controls",
    "17_observed_vs_predicted", "18_absolute_error_ecdf",
    "09_el_residual_vs_elevation", "07_mae_by_receiver",
    "12_selective_confidence_curve",
]
SOURCES = {
    "final_predictions.csv": "outputs/tables/final_predictions.csv",
    "final_metrics.csv": "outputs/tables/final_metrics.csv",
    "final_metrics_by_receiver.csv": "outputs/tables/final_metrics_by_receiver.csv",
    "dataset_report.json": "data/processed/aims_clean_features_report.json",
    "domain_classifier.csv": "deliverables/paper_figures/figure_sources/domain_shift_classifier_summary.csv",
    "domain_support.csv": "deliverables/paper_figures/figure_sources/domain_shift_support_summary.csv",
    "transfer_runs.csv": "deliverables/paper_figures/figure_sources/nested_fold_seed_metrics_saved.csv",
    "transfer_fold_means.csv": "deliverables/paper_figures/figure_sources/fold_means.csv",
    "transfer_bootstrap.csv": "deliverables/paper_figures/figure_sources/bootstrap_contrasts_saved.csv",
    "selective_curve.csv": "data/diagnostics/control/true_xel_selective_disagreement_thesis.csv",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value)


def numbers(rows, col):
    return np.array([float(row[col]) for row in rows])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(actual, expected, message, atol=5e-6):
    require(np.isclose(actual, expected, atol=atol, rtol=0),
            f"{message}: {actual} != {expected}")


def snapshot_sources(project, refresh):
    folder = HERE / "source_data"
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists() or refresh:
        paths = list(SOURCES.values()) + ["data/processed/aims_clean_features.csv"]
        for rel in paths:
            require((project / rel).is_file(), f"Missing project source: {project / rel}")
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"description": "Frozen plotting inputs; not a new model evaluation",
                    "originals": {}, "files": {}}
        for name, rel in SOURCES.items():
            shutil.copy2(project / rel, folder / name)
            manifest["originals"][name] = {"path": rel, "sha256": sha256(project / rel)}
        rel = "data/processed/aims_clean_features.csv"
        rows = read_csv(project / rel)
        fields = ["Timestamp", "Split", "Elevation", "Azimuth", "ReceiverCode",
                  "ReceiverBand", "target_XEL_arcsec", "target_EL_arcsec"]
        write_csv(folder / "dataset_events.csv", rows, fields)
        manifest["originals"]["dataset_events.csv"] = {
            "path": rel, "sha256": sha256(project / rel), "projection": fields}
        manifest["notes"] = [
            "Transfer summaries are saved six-decimal notebook-41 outputs, not raw predictions.",
            "Bootstrap intervals are imported, not recomputed from summary tables.",
            "Structural diagrams follow the revised chapter and local pipeline implementation.",
        ]
        for name in [*SOURCES, "dataset_events.csv"]:
            manifest["files"][name] = sha256(folder / name)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        require(sha256(folder / name) == digest, f"Modified snapshot: {name}")
    return folder


def metrics(rows):
    y = numbers(rows, "target")
    r = y - numbers(rows, "prediction")
    return {"MAE": np.mean(abs(r)), "RMSE": np.sqrt(np.mean(r ** 2)),
            "bias": np.mean(r), "p90_abs_error": np.quantile(abs(r), .9),
            "within_5_percent": 100 * np.mean(abs(r) <= 5),
            "R2": 1 - np.sum(r ** 2) / np.sum((y - np.mean(y)) ** 2)}


def load_and_validate(folder):
    data = {key: read_csv(folder / name) for key, name in [
        ("events", "dataset_events.csv"), ("predictions", "final_predictions.csv"),
        ("metrics", "final_metrics.csv"), ("receivers", "final_metrics_by_receiver.csv"),
        ("domain", "domain_classifier.csv"), ("support", "domain_support.csv"),
        ("runs", "transfer_runs.csv"), ("fold_means", "transfer_fold_means.csv"),
        ("bootstrap", "transfer_bootstrap.csv"), ("selective", "selective_curve.csv"),
    ]}
    events = {timestamp(row["Timestamp"]): row for row in data["events"]}
    require(len(events) == 668, "Dataset must contain 668 unique events")
    require(Counter(row["Split"] for row in events.values()) ==
            {"train": 400, "validation": 133, "test": 135}, "Unexpected split counts")
    report = json.loads((folder / "dataset_report.json").read_text(encoding="utf-8"))
    require(report["rows"] == 668 and report["columns"] - 8 == 43,
            "Dataset report disagrees with the stated sample/feature counts")
    require(report["skip_counts"]["thermal_jump"] == 197, "Window rejection count changed")
    require(list(events) == sorted(events), "Events are not chronological")
    require(len([k for k in read_csv(folder / "dataset_events.csv")[0]]) == 8,
            "Unexpected dataset snapshot schema")
    keys = set()
    for row in data["predictions"]:
        t = timestamp(row["Timestamp"])
        require(t in events, "Prediction timestamp missing from dataset")
        key = (t, row["axis"])
        require(key not in keys, f"Duplicate prediction: {key}")
        keys.add(key)
        require(row["axis"] in COLORS, "Unexpected prediction axis")
        row["split"] = events[t]["Split"]
        check_close(float(row["target"]), float(events[t][f'target_{row["axis"]}_arcsec']),
                    "Prediction/target alignment", atol=1e-5)
        require(row["ReceiverBand"] == events[t]["ReceiverBand"], "Receiver mismatch")
        check_close(float(row["residual"]), float(row["target"]) - float(row["prediction"]),
                    "Residual sign")
        check_close(float(row["abs_error"]), abs(float(row["residual"])), "Absolute residual")
    for row in data["metrics"]:
        subset = [p for p in data["predictions"]
                  if p["axis"] == row["axis"] and p["split"] == row["split"]]
        require(len(subset) == int(row["count"]), "Metric population mismatch")
        for key, value in metrics(subset).items():
            check_close(value, float(row[key]), f"{row['axis']}/{row['split']}/{key}")
    require(len(keys) == 2 * (133 + 135), "Unexpected prediction populations")
    data["test"] = [r for r in data["predictions"] if r["split"] == "test"]
    require(Counter(r["axis"] for r in data["test"]) == {"EL": 135, "XEL": 135},
            "Historical test selection is incorrect")
    for row in data["receivers"]:
        subset = [p for p in data["test"] if p["axis"] == row["axis"]
                  and p["ReceiverBand"] == row["ReceiverBand"]]
        require(len(subset) == int(row["count"]), "Receiver count mismatch")
        check_close(metrics(subset)["MAE"], float(row["MAE"]), "Receiver MAE")
    aliases = {"Ridge": "production_Ridge", "F1A": "F1A", "F2A": "F2A"}
    for row in data["fold_means"]:
        variant = aliases[row["model"]] if "model" in row else aliases[row["variant"]]
        subset = [r for r in data["runs"] if r["variant"] == variant
                  and int(r["fold"]) == int(row["fold"])]
        check_close(np.mean(numbers(subset, "MAE")), float(row["MAE_mean"]), "Fold mean")
    for row in data["bootstrap"]:
        require(float(row["ci95_low"]) <= float(row["delta_MAE"]) <= float(row["ci95_high"]),
                "Invalid saved bootstrap interval")
    require(len(data["domain"]) == 8 and len(data["support"]) == 3, "Domain table schema")
    require(all(0 <= float(r["AUC"]) <= 1 for r in data["domain"]), "Invalid AUC")
    require(sum(r["selection"] == "validation-selected" for r in data["selective"]) == 1,
            "Selective threshold missing or duplicated")
    return data


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True,
                         "savefig.facecolor": "white"})


def save(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, text in getattr(fig, "_checked_boxes", []):
        p, t = patch.get_window_extent(renderer), text.get_window_extent(renderer)
        require(p.x0 <= t.x0 and p.x1 >= t.x1 and p.y0 <= t.y0 and p.y1 >= t.y1,
                f"Text does not fit its box: {stem}: {text.get_text()}")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=.12,
                metadata={"Title": stem, "Subject": "Frozen thesis results; no model training"})
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    print(f"Rendered {stem}")


def diagram(title, height=4.6):
    fig, ax = plt.subplots(figsize=(7.8, height))
    fig.subplots_adjust(left=.01, right=.99, top=.89, bottom=.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=.97, color=INK)
    fig._checked_boxes = []
    return fig, ax


def box(ax, x, y, w, h, label, color=PALE_BLUE, size=9):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.014",
                          facecolor=color, edgecolor="#8B969F", linewidth=.8)
    ax.add_patch(patch)
    text = ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size,
                   color=INK, linespacing=1.4)
    ax.figure._checked_boxes.append((patch, text))
    return (x, y, w, h)


def arrow(ax, start, end, color=GREY):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": color,
                            "shrinkA": 3, "shrinkB": 3})


def note(ax, text, y=.03):
    ax.text(.5, y, text, ha="center", va="center", fontsize=8, color=GREY, linespacing=1.35)


def draw_audit():
    fig, ax = diagram("Coordinate and provenance audit", 4.3)
    box(ax, .02, .71, .28, .20, "Original cross-scan products\n868 mapped measurements")
    box(ax, .36, .71, .28, .20, "Control logs\nCommanded-offset state")
    box(ax, .70, .71, .28, .20, "Receiver and fit context\nSigma, flags, beam width")
    for x in [.16, .50, .84]:
        arrow(ax, (x, .70), (.5, .59))
    box(ax, .24, .38, .52, .21,
        'Historical "XEl Offset": longitudinal azimuth\n'
        r'$y_{\mathrm{XEL}}=\Delta A\cos(E),\qquad y_{\mathrm{EL}}=\Delta E$', PALE_TEAL)
    arrow(ax, (.5, .37), (.5, .27))
    box(ax, .16, .12, .68, .15, "Verified physical targets and sign provenance\n3 events with active commanded offsets excluded", PALE_ORANGE)
    note(ax, "Measured offsets are not automatically signed control commands.\nProjection is a coordinate correction, not a model improvement.", -.01)
    save(fig, STEMS[0])


def draw_data_pipeline(data):
    fig, ax = diagram("From cross-scans and telemetry to supervised samples", 5.0)
    box(ax, .05, .80, .39, .16, "868 audited cross-scans\nCoordinates, receiver, control state")
    box(ax, .56, .80, .39, .16, "Thermal telemetry\n295 channels on a 2-minute grid")
    arrow(ax, (.245, .79), (.245, .71))
    arrow(ax, (.755, .79), (.755, .71))
    box(ax, .05, .56, .39, .15, "Exclude 3 active-offset events\n865 target candidates", PALE_ORANGE)
    box(ax, .56, .56, .39, .15, "60 x 295 history windows\nGrid, finite-value and jump checks", PALE_ORANGE)
    arrow(ax, (.245, .55), (.5, .45))
    arrow(ax, (.755, .55), (.5, .45))
    box(ax, .23, .30, .54, .15, "197 thermal-window rejections\n668 retained labelled events", PALE_TEAL)
    for x in [.19, .5, .81]:
        arrow(ax, (.5, .29), (x, .21))
    for x, label in zip([.06, .37, .68], ["Train: 400", "Validation: 133", "Historical test: 135"]):
        box(ax, x, .10, .26, .11, label, "#F0F2F3")
    note(ax, "Chronological split precedes fitting statistical transformations.\nOnline use additionally requires an as-of timestamp-availability audit.", .01)
    save(fig, STEMS[1])


def draw_parallel():
    fig, ax = diagram("Final real-only prediction pipeline", 4.4)
    box(ax, .02, .39, .24, .29, "Thermal history\n60 x 295 samples\n+ geometry, time, receiver")
    box(ax, .33, .39, .25, .29, "Frozen feature builder\n43 predictors\nTraining-fitted PCA", PALE_TEAL)
    arrow(ax, (.27, .535), (.32, .535))
    box(ax, .68, .67, .29, .23, "Imputation + scaling\nHuber regression\nXEL estimate", PALE_BLUE)
    box(ax, .68, .18, .29, .23, "Imputation + scaling\nRidge regression\nEL estimate", PALE_ORANGE)
    arrow(ax, (.59, .59), (.67, .785), BLUE)
    arrow(ax, (.59, .48), (.67, .295), ORANGE)
    note(ax, "Independent axis-specific branches; no FEM is required at inference.\nOffline estimators: conversion to safe online commands remains a separate step.", .02)
    save(fig, STEMS[5])


def draw_temporal():
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    fig.subplots_adjust(left=.18, right=.98, top=.88, bottom=.32)
    ax.set_title("Chronological evaluation: separate roles for each partition", pad=16, fontweight="bold")
    palette = {"train": PALE_BLUE, "inner": "#E9DBAC", "outer": PALE_TEAL, "test": "#F7DDCE"}
    # Unit-width rectangles represent event positions, not elapsed wall-clock time.
    ax.broken_barh([(0, 400)], (4.5, .55), facecolors=palette["train"], edgecolors="white")
    ax.broken_barh([(400, 133)], (4.5, .55), facecolors=palette["inner"], edgecolors="white")
    ax.broken_barh([(533, 135)], (4.5, .55), facecolors=palette["test"], edgecolors="white")
    for x, label in [(200, "Train\n400"), (466.5, "Validation\n133"), (600.5, "Historical test\n135")]:
        ax.text(x, 4.775, label, ha="center", va="center", fontsize=8)
    for fold, n in enumerate([266, 355, 444], 1):
        y = 4.0 - fold
        # Inner splits are schematic; audited outer counts are exact.
        cut = n * .8
        ax.broken_barh([(0, cut)], (y, .55), facecolors=palette["train"], edgecolors="white")
        ax.broken_barh([(cut, n-cut)], (y, .55), facecolors=palette["inner"], edgecolors="white")
        ax.broken_barh([(n, 89)], (y, .55), facecolors=palette["outer"], edgecolors="white")
        ax.text(n/2, y+.275, f"Outer train: {n}", ha="center", va="center", fontsize=8)
        ax.text(n+44.5, y+.275, "Eval\n89", ha="center", va="center", fontsize=8)
        ax.text(610, y+.275, "Not used", ha="center", va="center", fontsize=8, color=GREY)
    ax.axvline(533, color=GREY, ls="--", lw=1)
    ax.set(xlim=(0, 668), ylim=(.4, 5.35), xlabel="Event index in chronological order (not elapsed time)")
    ax.set_xticks([0, 266, 355, 444, 533, 668])
    ax.set_yticks([4.775, 3.275, 2.275, 1.275], ["Historical split", "Outer fold 1", "Outer fold 2", "Outer fold 3"])
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(facecolor=palette[k], label=l) for k,l in [
        ("train", "Fitting population"), ("inner", "Validation / inner selection"),
        ("outer", "Outer evaluation"), ("test", "Historical test")]],
        loc="lower center", bbox_to_anchor=(.5, .09), ncol=2, frameon=False)
    fig.text(.5, .025, "Inner 80/20 selection is followed by an outer-train refit.\n"
             "Outer-evaluation targets do not select epochs or residual-correction strength.",
             ha="center", fontsize=8, color=GREY)
    save(fig, STEMS[3])


def draw_transfer_protocol():
    fig, ax = diagram("Matched encoder transfer: FEM v1 versus FEM v2", 5.5)
    box(ax, .23, .82, .54, .14, "Identical thermal windows and source-row keys\n30, 60, 85 degrees; 24,000 training windows", PALE_TEAL)
    for x in [.24, .76]:
        arrow(ax, (.5, .81), (x, .73))
    box(ax, .04, .57, .40, .16, "FEM v1 labels\nSupervised encoder pretraining", PALE_BLUE)
    box(ax, .56, .57, .40, .16, "FEM v2 common_only labels\nSupervised encoder pretraining", PALE_ORANGE)
    for x in [.24, .76]:
        arrow(ax, (x, .56), (x, .49))
    box(ax, .04, .31, .40, .18, "F1A: transfer encoder only\nDiscard source head\nFresh, seed-paired target head", PALE_BLUE)
    box(ax, .56, .31, .40, .18, "F2A: transfer encoder only\nDiscard source head\nFresh, seed-paired target head", PALE_ORANGE)
    for x in [.24, .76]:
        arrow(ax, (x, .30), (.5, .24))
    box(ax, .21, .09, .58, .15, "Fine-tune using real measured labels\nSame outer folds and seeds: 17, 29, 43", PALE_TEAL)
    note(ax, "Controls: random initialisation, real self-supervision, real-only Ridge.\nSynthetic labels are never substituted for measured real targets.", .005)
    save(fig, STEMS[6])


def tidy(ax):
    ax.grid(alpha=.25, lw=.6)


def timeline(data):
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 5.2), sharex=True, layout="constrained")
    colors = {"train": BLUE, "validation": "#B48A16", "test": ORANGE}
    boundaries = []
    for split in ["validation", "test"]:
        boundaries.append(min(timestamp(r["Timestamp"]) for r in data["events"] if r["Split"] == split))
    for ax, axis in zip(axes, ["XEL", "EL"]):
        for split, color in colors.items():
            rows = [r for r in data["events"] if r["Split"] == split]
            ax.scatter([timestamp(r["Timestamp"]) for r in rows], numbers(rows, f"target_{axis}_arcsec"),
                       s=12, alpha=.75, color=color, label=f"{split.capitalize()} (n={len(rows)})")
        ax.axhline(0, color=INK, lw=.7)
        for boundary in boundaries:
            ax.axvline(boundary, color=GREY, lw=.7, ls="--")
        ax.set_ylabel(f"{axis} target [arcsec]")
        tidy(ax)
    axes[0].legend(loc="upper left", ncol=3, frameon=True)
    axes[0].set_title("Real targets and the historical chronological split")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].set_xlabel("Time [UTC]")
    save(fig, STEMS[2])


def domain(data):
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.8, 6.0), layout="constrained")
    groups = ["Full reference", "Thermal level only", "Centered thermal shape", "Without IQR and elevation"]
    y = np.arange(4)
    for offset, classifier, color, marker in [(-.1, "Logistic regression", BLUE, "o"), (.1, "Random Forest", ORANGE, "s")]:
        vals = [float(next(r["AUC"] for r in data["domain"] if r["feature_set"] == g
                          and r["classifier"] == classifier)) for g in groups]
        a.scatter(vals, y+offset, color=color, marker=marker, s=35, label=classifier)
        for v, yy in zip(vals, y+offset):
            a.text(v-.008, yy, f"{v:.3f}", ha="right", va="center", fontsize=8)
    a.axvline(.5, color=GREY, ls="--", lw=1)
    a.set(xlim=(.48, 1.025), ylim=(3.5, -.55), xlabel="Domain-classifier ROC AUC")
    a.set_yticks(y, ["Full reference", "Thermal level only", "Centered thermal shape", "Without IQR\nand elevation"])
    a.set_title("(a) FEM-versus-real separability", loc="left")
    a.legend(loc="lower left", frameon=False)
    a.grid(axis="x", alpha=.25)
    groups = data["support"]
    y = np.arange(3)
    for shift, key, label, color in [(-.16, "within_fem_p95_percent", "Inside p95", BLUE),
                                    (.16, "within_fem_p99_percent", "Inside p99 (including p95)", TEAL)]:
        bars = b.barh(y+shift, numbers(groups, key), height=.28, color=color, label=label)
        b.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    b.set_yticks(y, [r["feature_set"] for r in groups])
    b.invert_yaxis()
    b.set_xlim(0, 36)
    b.set_xlabel("Real events within estimated FEM support [% of 668]")
    b.set_title("(b) Representation-dependent support coverage", loc="left")
    b.grid(axis="x", alpha=.25)
    b.legend(loc="upper right", frameon=False)
    save(fig, STEMS[4])


def transfer_controls(data):
    fig, (a,b) = plt.subplots(2, 1, figsize=(7.8, 6.7), layout="constrained")
    styles = {"production_Ridge": (INK, "s", "Real-only Ridge"),
              "F1A": (BLUE, "o", "FEM v1 encoder (F1A)"),
              "F2A": (ORANGE, "^", "FEM v2 encoder (F2A)")}
    for offset, (variant, (color, marker, label)) in zip([-.17, 0, .17], styles.items()):
        means = []
        for fold in [1, 2, 3]:
            rows = sorted([r for r in data["runs"] if r["variant"] == variant and int(r["fold"]) == fold],
                          key=lambda r: int(r["seed"]))
            vals = numbers(rows, "MAE")
            means.append(np.mean(vals))
            if len(vals) > 1:
                a.scatter(fold+offset+np.linspace(-.04, .04, len(vals)), vals,
                          s=28, marker=marker, facecolors="none", edgecolors=color, zorder=3)
        a.plot(np.arange(1, 4)+offset, means, color=color, marker=marker, label=label, lw=1.2)
    a.set_xticks([1,2,3], ["Fold 1\n266 train / 89 eval", "Fold 2\n355 train / 89 eval", "Fold 3\n444 train / 89 eval"])
    a.set(xlim=(.7,3.3), ylim=(5.6,12.2), ylabel="EL MAE [arcsec]")
    a.set_title("(a) Outer-fold MAE: filled = seed mean; open = individual seed", loc="left")
    a.legend(loc="upper left", frameon=False)
    a.grid(axis="y", alpha=.25)
    for yy, row, color in zip([2,1,0], data["bootstrap"], [ORANGE, ORANGE, BLUE]):
        d,lo,hi = [float(row[k]) for k in ["delta_MAE", "ci95_low", "ci95_high"]]
        b.errorbar(d, yy, xerr=[[d-lo],[hi-d]], fmt="o", color=color, capsize=4)
        b.text(hi+.07, yy, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8)
    b.axvline(0, color=INK, lw=1, ls="--")
    b.set_yticks([2,1,0], ["F2A - F1A", "F2A - Ridge", "Ridge + F1A residual\n- Ridge"])
    b.set(xlim=(-.75, 4.35), ylim=(-.55,2.55), xlabel="Delta EL MAE [arcsec]; negative favours candidate")
    b.set_title("(b) Day-block bootstrap, seed-averaged predictions: mean and 95% interval", loc="left")
    b.grid(axis="x", alpha=.25)
    save(fig, STEMS[7])


def scatter_predictions(data):
    fig, axes = plt.subplots(1,2,figsize=(7.8,3.8),layout="constrained")
    for ax, axis in zip(axes, ["XEL","EL"]):
        rows=[r for r in data["test"] if r["axis"] == axis]
        y,p=numbers(rows,"target"),numbers(rows,"prediction")
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max())
        margin=.07*(hi-lo)
        lo,hi=lo-margin,hi+margin
        ax.scatter(y,p,s=20,alpha=.75,color=COLORS[axis],edgecolor="white",linewidth=.3)
        ax.plot([lo,hi],[lo,hi],ls="--",color=INK,lw=1)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Observed offset [arcsec]",ylabel="Predicted offset [arcsec]")
        ax.set_aspect("equal",adjustable="box")
        ax.set_title(f"{axis}: historical test (n=135)")
        tidy(ax)
    save(fig,STEMS[8])


def ecdf(data):
    fig,axes=plt.subplots(1,2,figsize=(7.8,3.9),layout="constrained")
    for axis in ["XEL","EL"]:
        values=np.sort(numbers([r for r in data["test"] if r["axis"]==axis],"abs_error"))
        probability=100*np.arange(1,len(values)+1)/len(values)
        x=np.r_[0,values]
        y=np.r_[0,probability]
        for ax in axes:
            ax.step(x,y,where="post",color=COLORS[axis],label=axis)
    for ax in axes:
        ax.axvline(5,ls="--",color=TEAL,lw=1,label="5-arcsec reference")
        ax.set(xlabel="Absolute error [arcsec]",ylabel="Cumulative events [%]",ylim=(0,101))
        tidy(ax)
    axes[0].set(xlim=(0,20),title="(a) Central error range")
    axes[1].set(xlim=(0,95),ylim=(90,100.5),title="(b) Full tail, top 10%")
    axes[0].legend(loc="lower right",frameon=False)
    save(fig,STEMS[9])


def residual_elevation(data):
    fig,axes=plt.subplots(1,2,figsize=(7.8,4),layout="constrained")
    for ax,axis in zip(axes,["XEL","EL"]):
        rows=[r for r in data["test"] if r["axis"]==axis]
        points=ax.scatter(numbers(rows,"Elevation"),numbers(rows,"residual"),
                          c=numbers(rows,"abs_error"),cmap="viridis",s=20,alpha=.85)
        for ref in [-5,5]:ax.axhline(ref,ls="--",color=TEAL,lw=.8)
        ax.axhline(0,color=INK,lw=.8)
        ax.set(title=f"{axis}: historical test (n=135)",xlabel="Elevation [deg]",ylabel="Observed - predicted [arcsec]")
        fig.colorbar(points,ax=ax,label="Absolute error [arcsec]",shrink=.76,pad=.02)
        tidy(ax)
    save(fig,STEMS[10])


def receiver(data):
    fig,ax=plt.subplots(figsize=(7.8,4.2),layout="constrained")
    groups=["K","C_high","C_low"]
    for offset,axis in [(-.18,"XEL"),(.18,"EL")]:
        vals=[float(next(r["MAE"] for r in data["receivers"] if r["axis"]==axis and r["ReceiverBand"]==g)) for g in groups]
        bars=ax.bar(np.arange(3)+offset,vals,width=.34,color=COLORS[axis],label=axis)
        ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=9)
    ax.set_xticks(range(3),["K\nn = 109","C-high\nn = 11","C-low\nn = 15"])
    ax.axhline(5,color=TEAL,ls="--",label="5-arcsec MAE reference")
    ax.set(title="Historical-test MAE by receiver",ylabel="MAE [arcsec]",ylim=(0,11.7))
    ax.grid(axis="y",alpha=.25)
    ax.legend(ncol=3,loc="upper left",frameon=False)
    save(fig,STEMS[11])


def selective(data):
    rows=sorted(data["selective"],key=lambda r:float(r["coverage_percent"]))
    selected=next(r for r in rows if r["selection"]=="validation-selected")
    fig,ax=plt.subplots(figsize=(7.8,4.2),layout="constrained")
    ax.plot(numbers(rows,"coverage_percent"),numbers(rows,"MAE"),"o-",color=BLUE,label="Calibrated expert stack")
    ax.axhline(5,color=TEAL,ls="--",label="5-arcsec MAE reference")
    x,y=float(selected["coverage_percent"]),float(selected["MAE"])
    ax.scatter([x],[y],marker="*",s=140,color=ORANGE,zorder=5,label="Validation-selected threshold")
    ax.annotate("55 / 135 events retained\n40.74% coverage; MAE 3.916 arcsec",xy=(x,y),
                xytext=(48,3.55),fontsize=8,arrowprops={"arrowstyle":"-","color":GREY},color=INK)
    ax.set(xlim=(20,103),ylim=(3.25,5.65),xlabel="Retained historical-test events [%]",ylabel="XEL MAE [arcsec]",
           title="Expert disagreement and selective prediction (not the final Huber model)")
    ax.legend(loc="upper left",frameon=False)
    tidy(ax)
    save(fig,STEMS[12])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=HERE.parent)
    parser.add_argument("--refresh-sources",action="store_true")
    args=parser.parse_args()
    configure()
    folder=snapshot_sources(args.project_root.resolve(),args.refresh_sources)
    data=load_and_validate(folder)
    draw_audit()
    draw_data_pipeline(data)
    timeline(data)
    draw_temporal()
    domain(data)
    draw_parallel()
    draw_transfer_protocol()
    transfer_controls(data)
    scatter_predictions(data)
    ecdf(data)
    residual_elevation(data)
    receiver(data)
    selective(data)
    manifest={"figure_count":len(STEMS),"formats":["pdf","png"],"png_dpi":300,
              "source_manifest_sha256":sha256(folder/"manifest.json"),
              "matplotlib_version":matplotlib.__version__,"numpy_version":np.__version__,
              "no_training_or_inference":True,"checks":"All source hashes, split/target alignment and saved metrics passed",
              "figures":{s:{ext:sha256(HERE/f"{s}.{ext}") for ext in ["pdf","png"]} for s in STEMS}}
    (HERE/"generation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("PASS: 13 PDF + 13 PNG. No training, model inference, or new test comparison.")


if __name__=="__main__":
    main()
