"""Readable PNG graphs from the fixed, screened aggregate projection only.

This renderer never opens predictions, signals, cases or model weights. Keep
CSV/HTML and the audit manifest outside images/: the submission selection is
that PNG-only directory. An image is still a review candidate, not approval.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import struct
import textwrap

import numpy as np
import pandas as pd
from PIL import Image

from .export_review import METRICS, SCREENED, SUPPRESSED, number


@dataclass(frozen=True)
class Section:
    title: str
    labels: tuple
    metrics: tuple
    note: str
    groups: tuple = ()


PERFORMANCE = METRICS + ("n", "mothers", "positive_mothers", "negative_mothers", "positives")
PAIR = ("difference",)
SECTIONS = {
    "model_comparison": Section("Shared holdout: model performance", ("model",), PERFORMANCE,
        "Same holdout; models selected on validation data. AP depends on prevalence. CI: mother-cluster bootstrap.", ("scope",)),
    "paired_comparisons": Section("Paired model differences", ("left", "right"), PAIR,
        "Difference = LEFT minus RIGHT on the same cohort. CI crossing zero does not establish equivalence.", ("scope", "metric")),
    "cohort_counts": Section("Cohort composition", ("split",), ("n", "mothers", "positive_mothers", "negative_mothers", "positives"),
        "n = 5-minute segments. A mother can contribute both positive and negative segments; class mother counts need not sum to all mothers."),
    "cv_summary": Section("Development cohort: repeated grouped CV", ("model", "seed"), METRICS,
        "One point per seed/model OOF aggregate; the final holdout is excluded. Seed spread is not a confidence interval."),
    "validation_selection": Section("Validation: tree model selection", ("model", "seed"), ("validation_auprc",),
        "Selection uses validation AP, not test performance. All available candidates are shown."),
    "single_feature_selection": Section("Validation: single-feature candidates", ("feature",), ("validation_auroc", "validation_auprc"),
        "H2 selects the best single feature by validation AUROC. Spline models are nonlinear controls.", ("family",)),
    "seed_holdout_metrics": Section("Holdout: seed sensitivity", ("model", "seed"), METRICS + ("validation_auprc",),
        "Exploratory seed sensitivity. Do not select a seed using these holdout results; seed variation is not a CI."),
    "seed_paired_cat28_minus_comparator": Section("Seed-matched ablations", ("comparator", "seed"), ("auroc_difference", "auprc_difference", "brier_difference"),
        "Cat28 minus comparator, matched by seed. Positive AUROC/AP and negative Brier differences favor Cat28."),
    "feature_importance": Section("Global feature contribution", ("feature",), ("mean_abs_shap", "contributing_mothers"),
        "Mean absolute SHAP, screened on actual explanation contributors. Predictive contribution is not a causal effect."),
    "feature_response_train": Section("Training feature-response curves", ("feature",), ("positive_fraction",),
        "Training quantile bins; raw endpoints are omitted. Marginal association, not causality. One small bin suppresses the entire feature."),
    "feature_quality": Section("Feature measurement quality", ("feature", "measurement_status"), ("n", "mothers", "n_unique", "zero_fraction", "mean", "std"),
        "Mean/SD use each feature's native unit; magnitudes across features are not comparable. Zero does not necessarily mean clinical absence."),
    "preprocessing_sensitivity": Section("Sensitivity to 30-second smoothing", ("feature",), ("mean_abs_change",),
        "Mean absolute change in each feature's native unit; no cross-feature magnitude ranking is implied."),
    "figo_agreement": Section("Record-level FIGO annotation agreement", ("label", "feature"), ("n", "spearman", "mae", "auroc"),
        "Record-level annotations are compared with extracted summaries. This is not inter-rater reliability or segment-level ground truth.", ("label",)),
    "cnn_capacity_validation": Section("CNN capacity: validation candidates", ("width", "seed"),
        ("validation_auprc", "parameters", "best_epoch", "epochs_run", "seconds_this_invocation"),
        "Width is selected by mean validation AP across seeds. Invocation time may exclude earlier resumed training.", ("normalization",)),
    "cnn_selection": Section("Selected CNN configurations", ("model", "width", "seed"),
        ("validation_auprc", "parameters", "best_epoch", "epochs_run"),
        "Selected by validation AP. These configuration values are not test performance.", ("normalization",)),
    "cnn_capacity_summary": Section("CNN capacity: seed mean and spread", ("width",),
        ("mean_validation_auprc", "std_validation_auprc", "min_validation_auprc", "max_validation_auprc", "seeds"),
        "Mean, standard deviation and min/max across seeds; these spreads are not bootstrap confidence intervals.", ("normalization",)),
    "normalization_matched": Section("CNN normalization: matched comparison", ("arm", "width", "seed"), PAIR,
        "LEFT normalization minus RIGHT normalization at the same width/seed. CI crossing zero does not establish equivalence.",
        ("metric", "left_normalization", "right_normalization")),
    "inference_cost": Section("Measured inference cost", ("model", "device"),
        ("milliseconds_per_window", "median_microseconds_per_segment", "feature_extraction_smooth30_ms_per_segment",
         "feature_extraction_smooth0_ms_per_segment", "parameters", "model_file_bytes", "normalized_input_bytes",
         "cuda_peak_allocated_bytes", "windows", "segments", "seconds", "median_batch_seconds", "timing_repeats", "repeats"),
        "Timing scopes differ. Extraction is measured separately; adding separate timings is not measured end-to-end latency.", ("scope",)),
    "outcomes": Section("H5: outcome prediction", ("arm",), PERFORMANCE,
        "Observed outcomes only; mother-grouped OOF evaluation. Clinical plus reading uses expert reading summaries, not model predictions.", ("outcome",)),
    "outcome_associations": Section("H5: expert reading-outcome association", ("outcome",),
        ("spearman_abnormal_fraction", "expert_abnormal_fraction_auroc"),
        "Association alone is not incremental predictive value. Missing outcomes may bias the observed cohort."),
    "outcome_reading_cells": Section("H5: reading and outcome counts", ("reading_positive", "outcome_positive"), ("records", "mothers"),
        "Counts by expert reading and observed outcome (0 = negative, 1 = positive). Any small cell suppresses the entire contingency.", ("outcome",)),
    "outcome_missingness": Section("H5: observed versus missing outcomes", ("cohort",),
        ("records", "mothers", "reading_positive_records", "reading_negative_records", "reading_positive_mothers", "reading_negative_mothers"),
        "Missing/invalid outcomes are excluded from outcome models. Counts describe selection and missingness.", ("outcome",)),
    "outcome_cohort_descriptors": Section("H5: outcome-cohort descriptors", ("cohort",),
        ("mean", "median", "observed_records", "observed_mothers", "missing_records", "missing_mothers"),
        "Descriptors are screened on mothers actually observed for that variable. Means/medians are descriptive, not adjusted effects.", ("outcome", "feature")),
    "emr_matched_seeds": Section("EMR addition: matched seed differences", ("seed",), PAIR,
        "Waveform plus EMR minus waveform only, matched by seed. Positive AP/AUROC and negative Brier favor adding EMR.", ("metric",)),
    "emr_selection": Section("EMR addition: validation selection", ("arm", "selection_stage", "seed"),
        ("validation_auprc", "threshold"),
        "Equal candidate seeds/budget/selection rules in both arms. Operating thresholds are chosen on validation data."),
    "loso": Section("Leave-one-site-out evaluation", ("site", "arm"), PERFORMANCE + ("normal_records", "normal_records_alarmed"),
        "Site aliases Sxx. Platt calibration uses source validation only. Internal site exclusion is not independent external validation."),
    "official_reference": Section("Official models: reference evaluation", ("model",), PERFORMANCE,
        "Original training overlap is unknown. Emergency is a different task; YOLO comparisons use the common-image subset only.", ("scope",)),
    "hypothesis_evidence": Section("Research questions: paired evidence", ("question", "contrast", "evidence"), PAIR,
        "H2: AUROC; H3/H4: AP. Other questions require the domain figures. Inconclusive differences do not prove equivalence.", ("primary_metric",)),
}

DISPLAY = {
    "auroc": "AUROC (higher is better)", "auprc": "Average precision / AP (higher is better)",
    "prevalence": "Positive segment fraction", "sensitivity": "Sensitivity at validation threshold",
    "specificity": "Specificity at validation threshold", "ppv": "Positive predictive value",
    "f1": "F1 score", "brier": "Brier score (lower is better)",
    "normal_record_alarm_rate": "Normal-record alarm fraction",
    "false_alarms_per_normal_hour": "Positive 5-min windows / normal hour",
    "n": "Segments / observations (count)", "mean_abs_shap": "Mean |SHAP| (model output units)",
    "mean_abs_change": "Mean absolute change (feature-native unit)",
    "mean": "Mean (native unit)", "std": "Standard deviation (native unit)", "median": "Median (native unit)",
    "milliseconds_per_window": "Inference time (ms / window)",
    "median_microseconds_per_segment": "Inference time (microseconds / segment)",
    "feature_extraction_smooth30_ms_per_segment": "30s smoothing + extraction (ms / segment)",
    "feature_extraction_smooth0_ms_per_segment": "No smoothing + extraction (ms / segment)",
}
SHORT = {
    "cnn_small_channel_maxabs": "CNN small / channel maxabs", "cnn_medium_channel_maxabs": "CNN medium / channel maxabs",
    "cnn_small_per_segment_z": "CNN small / segment z", "cnn_medium_per_segment_z": "CNN medium / segment z",
    "shared_holdout": "Shared holdout", "channel_maxabs": "Channel maxabs", "per_segment_z": "Per-segment z",
    "common_image_subset_training_overlap_unknown": "Common-image subset; training overlap unknown",
    "Emergency_different_task_training_overlap_unknown": "Emergency task; training overlap unknown",
    "raw_window_normalization_and_inference_excludes_loading_interpolation": "Raw-window normalization + inference; loading/interpolation excluded",
    "feature_matrix_only_excludes_extraction_and_io": "Feature-matrix inference only; extraction and I/O excluded",
}
ROWS_PER_PAGE = 6
PANELS_PER_PAGE = 2
DPI = 200
IMAGE_NAME = re.compile(r"(?:00_(?:coverage|protocol)_p\d{3,}|\d{2}_(?:" + "|".join(SECTIONS) + r")_p\d{3,})\.png\Z")
COLOR = "#176889"
INK = "#173044"
MUTED = "#526473"


def _feature_unit(feature):
    if feature in ("fhr_mean", "fhr_min", "fhr_max", "fhr_sd", "stv", "decel_max_depth",
                   "figo_baseline", "figo_baseline_var", "hist_width", "hist_median", "hist_mode"):
        return "bpm"  # STV here is successive sampled FHR difference, not clinical STV.
    if feature in ("brady_frac", "tachy_frac", "decel_time_frac", "abnormal_fraction"):
        return "fraction"
    if feature in ("ft_corr0", "ft_corr_min"):
        return "correlation (unitless)"
    if feature == "ft_corr_min_lag_s":
        return "seconds"
    if feature in ("toco_mean", "toco_max", "toco_sd"):
        return "TOCO device units"
    if str(feature).startswith("n_"):
        return "events / 5-min window"
    return {"maternal_age": "years", "gestational_age": "weeks", "longest_abnormal_run": "consecutive 5-min windows"}.get(feature, "native unit")


def _human(value):
    return SHORT.get(str(value), str(value).replace("_", " "))


def _fmt(value):
    value = number(value)
    if value is None:
        return "NA"
    if abs(value) >= 1000 or value == int(value):
        return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"
    return f"{value:.4f}" if abs(value) >= .0001 or value == 0 else f"{value:.3g}"


def _cell(row, key):
    # Defense in depth: suppressed/onsite rows cannot become points or labels.
    status = row.get("review_status")
    if status is not None and status != SCREENED:
        return None
    return number(row.get(key))


def _labels(row, keys):
    parts = []
    for key in keys:
        value = row.get(key)
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        if isinstance(value, (int, float, np.number)):
            value = _cell(row, key)
            if value is None:
                continue
            parts.append(key.replace("_", " ") + "=" + _fmt(value))
        else:
            parts.append(_human(value))
    label = "\n".join(textwrap.wrap(" / ".join(parts), 29, break_long_words=True)) or "Unavailable"
    denominators = []
    for key, short in (("n", "N"), ("mothers", "M")):
        value = _cell(row, key)
        if value is not None:
            denominators.append(short + "=" + _fmt(value))
    return label + ("\n" + ", ".join(denominators) if denominators else "")


def _canvas(plt, title, subtitle, note, metadata, page, rows=8):
    # Explicit margins reserve space for multiline row labels and numeric CIs.
    fig = plt.figure(figsize=(20, max(8.6, 3.6 + rows * .78)), facecolor="white")
    fig.text(.03, .955, title, fontsize=21, weight="bold", color=INK, va="top")
    fig.text(.03, .905, textwrap.fill(subtitle, 150), fontsize=11, color=MUTED, va="top")
    profile = "MOCK: EXECUTION TEST ONLY" if metadata.get("profile") == "mock" else "PENDING INSTITUTION REVIEW"
    minimum = int(metadata.get("export_min_mothers", 10))
    footer = (f"{profile} | Page {page} | Minimum contributing mothers: {minimum} (screening, not approval).\n"
              "SUPPRESSED = screening failed; NA = unavailable/not computed/metric-specific suppression; neither means zero. N: observations; M: mothers.\n" + note)
    lines = []
    for line in footer.splitlines():
        lines.extend(textwrap.wrap(line, 175))
    fig.text(.03, .035, "\n".join(lines), fontsize=10, color=MUTED, va="bottom", linespacing=1.5)
    return fig


def _save(fig, path, plt):
    """Re-encode pixels only: no text, EXIF, attachments or hidden data chunks."""
    buffer = BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=DPI, facecolor="white")
        buffer.seek(0)
        with Image.open(buffer) as rendered:
            rgb = rendered.convert("RGB")
            pixels = Image.frombytes("RGB", rgb.size, rgb.tobytes())
            with BytesIO() as encoded:
                pixels.save(encoded, format="PNG", dpi=(DPI, DPI))
                payload = encoded.getvalue()
            if path.write_bytes(payload) != len(payload):
                raise OSError("Incomplete PNG write")
            pixels.close()
            rgb.close()
    finally:
        plt.close(fig)
        buffer.close()
    return path


def _interval(row, key):
    lo_key, hi_key = ("ci95_low", "ci95_high") if key == "difference" else (key + "_lo", key + "_hi")
    lo, hi = _cell(row, lo_key), _cell(row, hi_key)
    return (lo, hi) if lo is not None and hi is not None and lo <= hi else (None, None)


def _panel(axis, records, key, labels, *, metric=None):
    positions = np.arange(len(records))
    values = [_cell(row, key) for row in records]
    difference = key == "difference" or key.endswith("_difference")
    underlying = metric or (key.removesuffix("_difference") if difference else key)
    bounded = underlying in METRICS[:-1] or any(part in underlying for part in ("auroc", "auprc", "fraction"))
    title = DISPLAY.get(key, _human(key))
    if key in ("mean", "std", "median", "mean_abs_change", "mae") and len({row.get("feature") for row in records}) == 1:
        title = f"{_human(key)} ({_feature_unit(records[0].get('feature'))})"
    if difference:
        title = f"LEFT - RIGHT: {_human(underlying)}"
    axis.set_title(textwrap.fill(title, 40), fontsize=13, weight="bold", loc="left", pad=14)
    axis.set_yticks(positions, labels, fontsize=10)
    axis.set_ylim(len(records) - .5, -.5)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.tick_params(axis="x", labelsize=10)
    axis.grid(axis="x", color="#E3EAF0", linewidth=.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if difference:
        axis.axvline(0, color="#526473", ls="--", lw=1)
        direction = "Negative favors LEFT (lower Brier is better)" if underlying == "brier" else "Positive favors LEFT (higher AP/AUROC is better)"
        axis.set_xlabel(direction, fontsize=9)
    limits = []
    for pos, row, value in zip(positions, records, values):
        if value is None:
            label = "SUPPRESSED" if row.get("review_status") == SUPPRESSED else "NA"
            axis.text(1.04, pos, label, transform=axis.get_yaxis_transform(), va="center", fontsize=10, color=MUTED)
            continue
        lo, hi = _interval(row, key)
        limits.append(value)
        if lo is not None:
            limits.extend([lo, hi])
            axis.plot([lo, hi], [pos, pos], color=COLOR, lw=2)
            axis.plot([lo, hi], [pos, pos], "|", color=COLOR, markersize=8)
        axis.plot(value, pos, "o", color=COLOR, markersize=6, zorder=3)
        text = _fmt(value)
        if lo is not None:
            text += f"\n[{_fmt(lo)}, {_fmt(hi)}]"
        axis.text(1.04, pos, text, transform=axis.get_yaxis_transform(), va="center", fontsize=10, color=INK)
    if difference:
        limit = max([abs(v) for v in limits] + [.01]) * 1.18
        axis.set_xlim(-limit, limit)
    elif "spearman" in key:
        axis.set_xlim(-1.05, 1.05)
    elif bounded:
        axis.set_xlim(-.03, 1.03)
    elif limits:
        low, high = min(0, min(limits)), max(0, max(limits))
        pad = (high - low) * .12 or 1
        axis.set_xlim(low - pad, high + pad)
    else:
        axis.set_xlim(0, 1)
    axis.text(1.04, 1.035, "Value [95% CI]", transform=axis.transAxes, fontsize=9, color=MUTED)


def _groups(frame, keys):
    keys = [key for key in keys if key in frame]
    if not keys:
        return [("", frame)]
    result = []
    for values, part in frame.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        label = " | ".join(_human(key) + ": " + _human(value) for key, value in zip(keys, values))
        result.append((label, part))
    return result


def _generic_pages(plt, out, index, name, section, frame, metadata):
    files = []
    # Feature-native units must not share an axis across unlike quantities.
    keys = section.groups
    if name in ("feature_quality", "preprocessing_sensitivity"):
        # Quality counts/zero rates remain comparable; native-unit moments get their own pages below.
        ordinary = tuple(k for k in section.metrics if k not in ("mean", "std", "mean_abs_change"))
        native = tuple(k for k in section.metrics if k in ("mean", "std", "mean_abs_change"))
        specs = [(keys, ordinary), (("feature",), native)]
    else:
        specs = [(keys, section.metrics)]
    for group_keys, metrics in specs:
        if not metrics:
            continue
        for group_label, part in _groups(frame, group_keys):
            # Skip fields absent for this schema/group, but keep every present all-missing field explicit.
            present = [key for key in metrics if key in part]
            if not present:
                present = list(metrics[:1])
            records = part.to_dict("records")
            if not records:
                continue
            for start in range(0, len(records), ROWS_PER_PAGE):
                rows = records[start:start + ROWS_PER_PAGE]
                labels = [_labels(row, section.labels) for row in rows]
                for mi in range(0, len(present), PANELS_PER_PAGE):
                    metric_keys = present[mi:mi + PANELS_PER_PAGE]
                    page = len(files) + 1
                    subtitle = f"{name} | {group_label} | rows {start + 1}-{start + len(rows)} of {len(records)}"
                    fig = _canvas(plt, section.title, subtitle, section.note, metadata, page, len(rows))
                    for column, key in enumerate(metric_keys):
                        axis = fig.add_axes([.16 + column * .50, .24, .24, .57])
                        _panel(axis, rows, key, labels, metric=rows[0].get("metric"))
                    if len(metric_keys) == 1:
                        fig.text(.57, .68, "HOW TO READ", fontsize=14, weight="bold", color=INK)
                        fig.text(.57, .61, "\n".join(textwrap.wrap(section.note, 65)), fontsize=13, color=MUTED, va="top", linespacing=1.7)
                        if name == "hypothesis_evidence":
                            fig.text(.57, .35, "H1 / H5 / LOSO: see the corresponding domain figures.\nNo automatic hypothesis acceptance.", fontsize=12, color=MUTED)
                    files.append(_save(fig, out / f"{index:02d}_{name}_p{page:03d}.png", plt))
    return files


def _response_pages(plt, out, index, frame, metadata):
    files = []
    section = SECTIONS["feature_response_train"]
    for page, start in enumerate(range(0, frame.feature.nunique(), 4), 1):
        groups = list(frame.groupby("feature", sort=True))[start:start + 4]
        fig = _canvas(plt, section.title, "All available features; bin labels retain screened sample sizes", section.note, metadata, page, rows=14)
        for panel, (feature, part) in enumerate(groups):
            axis = fig.add_axes([.075 + (panel % 2) * .50, .58 - (panel // 2) * .33, .38, .22])
            rows = part.sort_values("bin").to_dict("records")
            valid = [row for row in rows if _cell(row, "bin") is not None and _cell(row, "positive_fraction") is not None]
            axis.set_title(_human(feature), fontsize=14, weight="bold", loc="left")
            axis.set(ylim=(-.03, 1.08), ylabel="Abnormal segment fraction")
            axis.grid(color="#E3EAF0", linewidth=.7)
            if not valid:
                axis.text(.5, .5, "SUPPRESSED / UNAVAILABLE", ha="center", va="center", transform=axis.transAxes, color=MUTED)
                axis.set_xticks([])
                continue
            # Retain gaps; never connect across an unavailable bin.
            x = [(_cell(row, "bin") if _cell(row, "bin") is not None else np.nan) for row in rows]
            y = [(_cell(row, "positive_fraction") if _cell(row, "positive_fraction") is not None else np.nan) for row in rows]
            axis.plot(x, y, "o-", color=COLOR, lw=1.8)
            labels = []
            for row in valid:
                label = f"Bin {_fmt(_cell(row, 'bin'))}\nM={_fmt(_cell(row, 'n_mothers'))} / N={_fmt(_cell(row, 'n_segments'))}"
                labels.append(label)
                value = _cell(row, "positive_fraction")
                axis.annotate(_fmt(value), (_cell(row, "bin"), value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
            axis.set_xticks([_cell(row, "bin") for row in valid], labels, fontsize=9)
            axis.set_xlabel("M: distinct mothers; N: segments. No raw bin endpoints.", fontsize=9)
        files.append(_save(fig, out / f"{index:02d}_feature_response_train_p{page:03d}.png", plt))
    return files


def _coverage(plt, out, tables, metadata):
    files = []
    names = list(SECTIONS)
    for start in range(0, len(names), 10):
        subset = names[start:start + 10]
        page = len(files) + 1
        fig = _canvas(plt, "Image atlas: analysis coverage", "Counts below are analysis rows, NOT patient counts. Empty analyses are retained here.",
                      "Only the PNG files in images/ are image submission candidates. Review CSV/HTML/JSON stay outside that folder. All images require institutional review.", metadata, page, 10)
        axis = fig.add_axes([.29, .23, .47, .57])
        axis.set_yticks(range(len(subset)), [_human(n) for n in subset], fontsize=12)
        axis.invert_yaxis()
        for i, name in enumerate(subset):
            frame = tables.get(name, pd.DataFrame())
            available, suppressed = 0, 0
            for row in frame.to_dict("records"):
                if row.get("review_status") == SUPPRESSED:
                    suppressed += 1
                elif any(_cell(row, key) is not None for key in SECTIONS[name].metrics):
                    available += 1
            missing = len(frame) - available - suppressed
            left = 0
            for count, color in ((available, COLOR), (suppressed, "#B4BEC7"), (missing, "#E4EAF0")):
                axis.barh(i, count, left=left, color=color, height=.55)
                left += count
            axis.text(1.025, i, f"{available} available / {suppressed} suppressed / {missing} NA" if len(frame) else "NOT AVAILABLE / NOT RUN",
                      transform=axis.get_yaxis_transform(), va="center", fontsize=11, color=MUTED)
        axis.set_xlabel("Number of aggregate analysis rows", fontsize=12)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", color="#E3EAF0")
        files.append(_save(fig, out / f"00_coverage_p{page:03d}.png", plt))
    return files


def _protocol_pages(plt, out, metadata):
    """Plot the allowlisted run settings; provenance text contains no input hashes."""
    keys = ("cv_folds", "tree_iterations", "tree_patience", "cnn_epochs", "cnn_patience", "bootstrap", "batch_size")
    files = []
    for start in range(0, len(keys), 4):
        page = len(files) + 1
        fig = _canvas(plt, "Analysis protocol and computation budget", "Settings fixed before evaluation; these are configuration values, not study findings.",
                      "Primary metric: AP (H1/H4); AUROC (H2). Splitting and bootstrap unit: mother. H4 has no equivalence/noninferiority claim.", metadata, page, 10)
        for i, key in enumerate(keys[start:start + 4]):
            axis = fig.add_axes([.07 + (i % 2) * .27, .61 - (i // 2) * .27, .20, .13])
            value = number(metadata.get(key))
            axis.set_title(_human(key), fontsize=13, weight="bold", loc="left")
            if value is not None:
                axis.barh([0], [value], color=COLOR, height=.5)
                axis.text(.02, .85, _fmt(value), transform=axis.transAxes, fontsize=12, weight="bold")
                axis.set_xlim(0, max(value * 1.15, 1))
            else:
                axis.text(.5, .5, "NOT AVAILABLE", ha="center", va="center", transform=axis.transAxes)
            axis.set_yticks([])
            axis.spines[["top", "right", "left"]].set_visible(False)
        seeds = [number(v) for v in metadata.get("seeds", []) if number(v) is not None]
        history = metadata.get("cohort_history")
        history = history if history in ("checked", "not_checked") else "not_checked"
        notes = ["CONFIGURATION / PROVENANCE", "Seeds: " + (", ".join(_fmt(v) for v in seeds) or "not available"),
                 "Prior-cohort overlap: " + history, "AP is average precision; prevalence matters.",
                 "Intervals reflect mother-cluster resampling.", "Point estimates are rounded to 4 decimals.",
                 "Seed SD/min/max are not 95% CIs."]
        digest = metadata.get("package_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            notes += ["Analysis code package SHA256:", digest[:32], digest[32:]]
        fig.text(.66, .77, "\n\n".join(notes), fontsize=12, va="top", color=MUTED)
        files.append(_save(fig, out / f"00_protocol_p{page:03d}.png", plt))
    return files


def write_image_atlas(out, tables, metadata):
    """Call only with collect_tables(..., screened=True); never with onsite tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .export_review import EXPORT_TABLES

    if set(SECTIONS) != EXPORT_TABLES or set(tables) - EXPORT_TABLES:
        raise ValueError("Image atlas and screened export schema differ")
    for name, frame in tables.items():
        if "review_status" in frame and not frame.review_status.isin([SCREENED, SUPPRESSED]).all():
            raise ValueError("Images require screened aggregate tables")
        if name != "hypothesis_evidence" and not frame.empty and "review_status" not in frame:
            raise ValueError("Images require screening status")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Image atlas destination must be empty")
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 100,
                         "savefig.bbox": None, "axes.labelcolor": INK, "text.color": INK}):
        files = _coverage(plt, out, tables, metadata) + _protocol_pages(plt, out, metadata)
        for index, (name, section) in enumerate(SECTIONS.items(), 1):
            frame = tables.get(name, pd.DataFrame())
            if frame.empty:
                continue
            if name == "feature_response_train" and {"feature", "bin", "positive_fraction"}.issubset(frame):
                files += _response_pages(plt, out, index, frame, metadata)
            else:
                files += _generic_pages(plt, out, index, name, section, frame, metadata)
    validate_image_directory(out)
    return files


def validate_image_directory(out):
    """Reject sidecars, unknown names, metadata payloads, invalid PNGs and symlinks."""
    root = Path(out)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Symbolic-link image directory is not accepted")
    paths = sorted(root.iterdir())
    if not paths or not any(p.name.startswith("00_coverage_") for p in paths):
        raise ValueError("Image atlas coverage figures are missing")
    for path in paths:
        if path.is_symlink() or not path.is_file() or not IMAGE_NAME.fullmatch(path.name):
            raise ValueError("Unexpected file in PNG-only image directory")
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Invalid PNG signature")
        offset, ended = 8, False
        while offset + 12 <= len(data):
            size = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            if kind not in (b"IHDR", b"IDAT", b"IEND", b"pHYs"):
                raise ValueError("PNG contains unexpected metadata/payload")
            offset += size + 12
            if kind == b"IEND":
                ended = True
                break
        if not ended or offset != len(data):
            raise ValueError("Invalid PNG or trailing payload")
        with Image.open(path) as img:
            if img.mode != "RGB" or min(img.size) < 1000:
                raise ValueError("Unexpected PNG format/resolution")
            img.verify()
    return {"status": "validated_pending_institution_review", "png_files": len(paths)}
