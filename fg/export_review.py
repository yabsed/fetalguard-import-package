"""Construct an aggregate-only review package; this is NOT export approval.

No files are copied from the run. Every output is rebuilt from a fixed schema.
Source identifiers are used only in memory to count distinct mothers. Unknown
strings/keys, individual predictions, raw extrema, paths and weights never pass
through. Small/unknown denominators are suppressed, not merely annotated.
"""
import hashlib
import html
import math
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json, write_json
from .features import CAT28, GROUPS


METRICS = ("auroc", "auprc", "prevalence", "sensitivity", "specificity", "ppv", "f1", "brier",
           "normal_record_alarm_rate", "false_alarms_per_normal_hour")
MODELS = {"cat28", "cat18", "xgb28", "robust", "logistic28", "best_single", "best_single_nonlinear", "cat28_unsmoothed",
          "cat28_emr", "clinical", "clinical_plus_reading", "official_yolo", "cat28_same_image_subset",
          "official_xgb_original", "official_xgb_corrected"} | {"without_" + g for g in GROUPS} | {
          "cnn_" + size + "_" + norm for size in ("small", "medium") for norm in ("channel_maxabs", "per_segment_z")}
OUTCOMES = {"ph_lt_7_20", "apgar1_lt_7", "apgar5_lt_7"}
NORMS = {"channel_maxabs", "per_segment_z"}
CV_MODELS = MODELS | {prefix + feature for prefix in ("single_", "nonlinear_single_") for feature in CAT28}
FIGO_LABELS = {"BaseLine", "Baseline_Variability", "Acceleration", "Early_deceleration", "Late_deceleration", "Variable_deceleration", "Prolonged_deceleration"}
STATUSES = {"ok", "complete", "disabled_by_config", "missing_field", "insufficient_classes", "insufficient_outcome_groups",
            "insufficient_classes_in_grouped_fold", "constant", "structurally_unobservable", "not_tested", "unavailable"}
COUNTS = ("n", "mothers", "positive_mothers", "negative_mothers", "positives")
SCREENED = "screened_aggregate_pending_review"
SUPPRESSED = "suppressed_requires_review"


def number(value):
    """JSON numbers / parsed CSV numeric values only; never stringify objects."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return float(value) if math.isfinite(float(value)) else None
    return None


def source(run, relative):
    root = Path(run)
    path = root / relative
    if path.is_symlink() or root.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
        raise ValueError("Symbolic-link sources are not accepted for aggregate export")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source escapes the internal run")
    return path


def load_json(run, relative, default=None):
    path = source(run, relative)
    return read_json(path) if path.is_file() else ({} if default is None else default)


def load_table(run, relative):
    path = source(run, relative)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"record_id": str, "mother_id": str, "site": str}, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def cohort_counts(frame):
    if frame.empty or not {"mother_id", "target"}.issubset(frame):
        return {}
    counts = {"n": len(frame), "mothers": frame.mother_id.nunique(),
            "positive_mothers": frame.loc[frame.target == 1, "mother_id"].nunique(),
            "negative_mothers": frame.loc[frame.target == 0, "mother_id"].nunique(),
            "positives": int(frame.target.sum())}
    if {"record_all_normal", "score", "threshold"}.issubset(frame):
        normal = frame.loc[frame.record_all_normal == 1].assign(alarm=lambda f: f.score >= f.threshold)
        counts["normal_mothers"] = normal.mother_id.nunique()
        counts["normal_alarm_mothers"] = normal.loc[normal.alarm, "mother_id"].nunique()
        counts["normal_quiet_mothers"] = counts["normal_mothers"] - counts["normal_alarm_mothers"]
    return counts


def adequate(counts, minimum, class_counts=True):
    fields = ("mothers", "positive_mothers", "negative_mothers") if class_counts else ("mothers",)
    return all(number(counts.get(k)) is not None and number(counts[k]) >= minimum for k in fields)


def screen_row(labels, values, counts, minimum, *, class_counts=True, extra_cells=(), screened=True):
    allowed = not screened or (adequate(counts, minimum, class_counts) and all(number(v) is not None and number(v) >= minimum for v in extra_cells))
    row = dict(labels, review_status=(SCREENED if allowed else SUPPRESSED) if screened else "onsite_only_not_screened")
    for key, value in values.items():
        row[key] = number(value) if allowed else None
    return row


def metric_rows(label, result, counts, minimum, screened=True):
    if not isinstance(result, dict):
        return []
    counts = dict({k: result.get(k) for k in COUNTS}, **counts)
    point, cis = result.get("point", {}), result.get("ci95", {})
    values = {k: counts.get(k) for k in COUNTS}
    values.update({k: point.get(k) for k in METRICS})
    for key in METRICS:
        ci = cis.get(key)
        values[key + "_lo"], values[key + "_hi"] = ci if isinstance(ci, list) and len(ci) == 2 else (None, None)
    row = screen_row(label, values, counts, minimum, screened=screened)
    if screened and not all(number(counts.get(k)) is not None and number(counts[k]) >= minimum
                            for k in ("normal_mothers", "normal_alarm_mothers", "normal_quiet_mothers")):
        for key in ("normal_record_alarm_rate", "false_alarms_per_normal_hour"):
            for suffix in ("", "_lo", "_hi"):
                row[key + suffix] = None
    return [row]


def paired_rows(label, result, counts, minimum, screened=True):
    rows = []
    if not isinstance(result, dict):
        return rows
    for metric in ("auroc", "auprc", "brier"):
        value = result.get("metrics", {}).get(metric)
        if not isinstance(value, dict):
            continue
        ci = value.get("ci95")
        lo, hi = ci if isinstance(ci, list) and len(ci) == 2 else (None, None)
        rows.append(screen_row(dict(label, metric=metric), dict(difference=value.get("difference"), ci95_low=lo, ci95_high=hi),
                               counts, minimum, screened=screened))
    return rows


def _split_frames(run):
    segments, split = load_table(run, "data/segments.csv"), load_table(run, "splits/segments.csv")
    if not segments.empty and not split.empty and {"record_id", "seg_idx", "split"}.issubset(split):
        if "split" not in segments:
            segments = segments.merge(split[["record_id", "seg_idx", "split"]], on=["record_id", "seg_idx"], validate="one_to_one")
        return {name: segments.loc[segments.split == name] for name in ("train", "val", "test")}
    return {name: pd.DataFrame() for name in ("train", "val", "test")}


def collect_tables(run, cfg, screened=True):
    """Fixed-schema projection shared by onsite and review reports."""
    minimum = max(1, int(cfg.get("export_min_mothers", 10)))
    tables = {}
    split = _split_frames(run)
    counts = {name: cohort_counts(frame) for name, frame in split.items()}
    predictions = load_table(run, "experiment_a/test_predictions.csv")
    baseline = predictions.loc[predictions.model == "cat28"] if "model" in predictions else pd.DataFrame()
    test_counts = counts["test"] or cohort_counts(baseline)
    dev = pd.concat([split["train"], split["val"]], ignore_index=True)
    dev_counts = cohort_counts(dev)
    models = load_json(run, "experiment_a/holdout_metrics.json")
    models.update(load_json(run, "experiment_b/metrics.json"))
    cnn_predictions = load_table(run, "experiment_b/test_predictions.csv")
    all_predictions = pd.concat([predictions, cnn_predictions], ignore_index=True)
    rows, pairs = [], []
    for model in sorted(MODELS & models.keys()):
        result = models[model]
        model_frame = all_predictions.loc[all_predictions.model == model] if "model" in all_predictions else pd.DataFrame()
        rows += metric_rows({"model": model, "scope": "shared_holdout"}, result, cohort_counts(model_frame) or test_counts, minimum, screened)
    for file in ("experiment_a/paired_cat28_minus_comparator.json", "experiment_b/paired_cat28_minus_cnn.json"):
        raw = load_json(run, file)
        for model in sorted(MODELS & raw.keys()):
            pairs += paired_rows({"left": "cat28", "right": model, "scope": "shared_holdout"}, raw[model], test_counts, minimum, screened)
    tables["model_comparison"] = pd.DataFrame(rows)
    tables["paired_comparisons"] = pd.DataFrame(pairs)
    tables["cohort_counts"] = pd.DataFrame([screen_row({"split": name}, {k: count.get(k) for k in COUNTS}, count, minimum, screened=screened)
                                           for name, count in counts.items()])

    # CV / seed / selection tables: only fixed model/feature identifiers and numeric columns.
    for filename, cohort, labels, numerics in (
        ("cv_summary", dev_counts, ("model",), ("seed",) + METRICS),
        ("validation_selection", counts["val"], ("model",), ("seed", "validation_auprc")),
        ("single_feature_selection", counts["val"], ("feature", "family"), ("validation_auroc", "validation_auprc")),
        ("seed_holdout_metrics", test_counts, ("model",), ("seed", "validation_auprc") + METRICS),
        ("seed_paired_cat28_minus_comparator", test_counts, ("comparator",), ("seed", "auroc_difference", "auprc_difference", "brier_difference")),
    ):
        raw = load_table(run, "experiment_a/" + filename + ".csv")
        valid_rows = []
        for row in raw.to_dict("records"):
            if any((key in ("model", "comparator") and row.get(key) not in CV_MODELS) or
                   (key == "feature" and row.get(key) not in CAT28) or
                   (key == "family" and row.get(key) not in ("linear", "nonlinear_spline", None)) for key in labels):
                continue
            names = {key: row[key] for key in labels if key in row and (key != "family" or row[key] in ("linear", "nonlinear_spline"))}
            projected = screen_row(names, {k: row.get(k) for k in numerics}, cohort, minimum, screened=screened)
            if screened and filename in ("cv_summary", "seed_holdout_metrics"):
                # These aggregate source rows do not identify the contributing
                # normal/alarmed/quiet mothers. Overall class counts cannot
                # establish the alarm-specific minimum cell sizes.
                projected["normal_record_alarm_rate"] = None
                projected["false_alarms_per_normal_hour"] = None
            valid_rows.append(projected)
        tables[filename] = pd.DataFrame(valid_rows)
    rows = []
    explanations = load_table(run, "experiment_a/local_explanations.csv")
    explained_counts = {}
    explained_pool = split["test"] if not split["test"].empty else baseline
    keys = ["record_id", "seg_idx"]
    if set(keys).issubset(explanations) and set(keys + ["mother_id", "target"]).issubset(explained_pool):
        contributors = explanations[keys].merge(explained_pool[keys + ["mother_id", "target"]], on=keys, how="left", validate="one_to_one")
        if not contributors[["mother_id", "target"]].isna().any().any():
            explained_counts = cohort_counts(contributors)
    for row in load_table(run, "experiment_a/feature_importance.csv").to_dict("records"):
        if row.get("feature") in CAT28:
            rows.append(screen_row({"feature": row["feature"]}, {"mean_abs_shap": row.get("mean_abs_shap"),
                                   "contributing_mothers": explained_counts.get("mothers")}, explained_counts, minimum, screened=screened))
    tables["feature_importance"] = pd.DataFrame(rows)
    rows = []
    for row in load_table(run, "experiment_a/feature_response_train.csv").to_dict("records"):
        if row.get("feature") not in CAT28:
            continue
        c = {"mothers": row.get("n_mothers"), "positive_mothers": row.get("positive_mothers"), "negative_mothers": row.get("negative_mothers")}
        # Bins are numbered; raw endpoint/extreme values are intentionally excluded.
        rows.append(screen_row({"feature": row["feature"], "scope": "training_only"},
                    {k: row.get(k) for k in ("bin", "n_segments", "n_mothers", "n_positive", "positive_mothers", "negative_mothers", "positive_fraction")},
                    c, minimum, screened=screened))
    # Suppress every bin of a feature when one fails: prevents differencing via the train totals.
    if screened:
        blocked = {r["feature"] for r in rows if r["review_status"] == SUPPRESSED}
        for row in rows:
            if row["feature"] in blocked:
                for key in row.keys() - {"feature", "scope", "review_status"}:
                    row[key] = None
                row["review_status"] = SUPPRESSED
    tables["feature_response_train"] = pd.DataFrame(rows)
    rows = []
    for row in load_table(run, "data/feature_quality.csv").to_dict("records"):
        if row.get("feature") not in CAT28:
            continue
        label = {"feature": row["feature"], "measurement_status": row.get("status") if row.get("status") in STATUSES else "unavailable"}
        row_counts = {k: row.get(k) for k in COUNTS}
        projected = screen_row(label, {k: row.get(k) for k in ("n", "mothers", "n_unique", "zero_fraction", "mean", "std")},
                               row_counts, minimum, screened=screened)
        if screened and not all(number(row.get(k)) is not None and (number(row[k]) == 0 or number(row[k]) >= minimum)
                                for k in ("zero_mothers", "nonzero_mothers")):
            # A mean/std with only one nonzero mother can disclose that mother's
            # contribution even if the zero fraction itself has been blanked.
            for key in ("n", "mothers", "n_unique", "zero_fraction", "mean", "std"):
                projected[key] = None
            projected["review_status"] = SUPPRESSED
        rows.append(projected)
    tables["feature_quality"] = pd.DataFrame(rows)
    quality_source = load_table(run, "data/feature_quality.csv")
    quality_by_feature = {row["feature"]: row for row in quality_source.to_dict("records") if row.get("feature") in CAT28}
    rows = []
    for row in load_table(run, "data/preprocessing_sensitivity.csv").to_dict("records"):
        if row.get("feature") in CAT28:
            rows.append(screen_row({"feature": row["feature"]}, {"mean_abs_change": row.get("mean_abs_change")},
                                  quality_by_feature.get(row["feature"], {}), minimum, screened=screened))
    tables["preprocessing_sensitivity"] = pd.DataFrame(rows)
    agreement = load_table(run, "supplementary/figo_annotation_agreement.csv")
    annotation_pairs = load_table(run, "supplementary/figo_record_pairs.csv")
    records = load_table(run, "data/records.csv")
    if not annotation_pairs.empty and {"record_id", "mother_id"}.issubset(records):
        annotation_pairs = annotation_pairs.merge(records[["record_id", "mother_id"]], on="record_id", validate="many_to_one")
    rows = []
    for row in agreement.to_dict("records"):
        if row.get("label") not in FIGO_LABELS or row.get("feature") not in CAT28:
            continue
        sub = annotation_pairs.loc[annotation_pairs.label == row["label"]] if "label" in annotation_pairs else pd.DataFrame()
        c = {"mothers": sub.mother_id.nunique()} if "mother_id" in sub else {}
        extra = []
        if row["label"] != "BaseLine" and {"label_value", "mother_id"}.issubset(sub):
            extra = sub.groupby("label_value").mother_id.nunique().tolist()
        rows.append(screen_row({"label": row["label"], "feature": row["feature"]},
                    {k: row.get(k) for k in ("n", "spearman", "mae", "auroc")}, c, minimum,
                    class_counts=False, extra_cells=extra, screened=screened))
    tables["figo_agreement"] = pd.DataFrame(rows)

    rows = []
    for row in load_table(run, "experiment_b/capacity_validation.csv").to_dict("records"):
        if row.get("normalization") not in NORMS:
            continue
        rows.append(screen_row({"normalization": row["normalization"]}, {k: row.get(k) for k in
                    ("width", "seed", "parameters", "best_epoch", "epochs_run", "validation_auprc", "seconds_this_invocation")}, counts["val"], minimum, screened=screened))
    tables["cnn_capacity_validation"] = pd.DataFrame(rows)
    selections = load_json(run, "experiment_b/selection.json", [])
    rows = []
    for row in selections if isinstance(selections, list) else []:
        if row.get("model") in MODELS and row.get("normalization") in NORMS:
            rows.append(screen_row({"model": row["model"], "normalization": row["normalization"]},
                        {k: row.get(k) for k in ("width", "seed", "parameters", "best_epoch", "epochs_run", "validation_auprc")}, counts["val"], minimum, screened=screened))
    tables["cnn_selection"] = pd.DataFrame(rows)
    rows = []
    for row in load_table(run, "experiment_b/capacity_summary.csv").to_dict("records"):
        if row.get("normalization") in NORMS:
            rows.append(screen_row({"normalization": row["normalization"]}, {k: row.get(k) for k in
                        ("width", "seeds", "mean_validation_auprc", "std_validation_auprc", "min_validation_auprc", "max_validation_auprc")},
                        counts["val"], minimum, screened=screened))
    tables["cnn_capacity_summary"] = pd.DataFrame(rows)
    matched = load_json(run, "experiment_b/normalization_matched.json")
    matched_rows = []
    for row in matched.get("comparisons", []):
        if row.get("arm") not in ("small", "medium") or row.get("left_normalization") not in NORMS or row.get("right_normalization") not in NORMS:
            continue
        label = {"arm": row["arm"], "left_normalization": row["left_normalization"], "right_normalization": row["right_normalization"]}
        for pair in paired_rows(label, row.get("paired_difference", {}), test_counts, minimum, screened):
            for key in ("width", "seed"):
                pair[key] = number(row.get(key))
            matched_rows.append(pair)
    tables["normalization_matched"] = pd.DataFrame(matched_rows)
    cost_rows = []
    extraction_cost = load_json(run, "data/summary.json").get("feature_extraction_ms_per_segment", {})
    for row in load_table(run, "experiment_b/inference_cost.csv").to_dict("records"):
        if row.get("model") in MODELS:
            label = {"model": row["model"], "scope": "raw_window_normalization_and_inference_excludes_loading_interpolation"}
            if row.get("device") in ("cpu", "cuda"):
                label["device"] = row["device"]
            cost_rows.append(screen_row(label, {k: row.get(k) for k in ("width", "seed", "parameters", "model_file_bytes", "windows", "seconds", "milliseconds_per_window", "normalized_input_bytes", "cuda_peak_allocated_bytes", "timing_repeats")}, test_counts, minimum, screened=screened))
    for model in sorted(MODELS & models.keys()):
        timing = models[model].get("inference_timing", {})
        if timing:
            values = {k: timing.get(k) for k in ("median_batch_seconds", "segments", "median_microseconds_per_segment", "repeats")}
            values["model_file_bytes"] = models[model].get("model_bytes")
            values["feature_extraction_smooth30_ms_per_segment"] = extraction_cost.get("smooth30")
            values["feature_extraction_smooth0_ms_per_segment"] = extraction_cost.get("smooth0")
            cost_rows.append(screen_row({"model": model, "scope": "feature_matrix_only_excludes_extraction_and_io"}, values, test_counts, minimum, screened=screened))
    tables["inference_cost"] = pd.DataFrame(cost_rows)

    outcomes = load_json(run, "supplementary/outcomes.json")
    outcome_pred = load_table(run, "supplementary/outcome_oof_predictions.csv")
    outcome_rows, associations, cells, missingness, descriptors = [], [], [], [], []
    for name in sorted(OUTCOMES & outcomes.keys()):
        result = outcomes[name]
        subset = outcome_pred.loc[(outcome_pred.outcome == name) & (outcome_pred.arm == "clinical")] if {"outcome", "arm"}.issubset(outcome_pred) else pd.DataFrame()
        c = cohort_counts(subset)
        if not c:
            c = {k: result.get(k, result.get("metrics", {}).get("clinical", {}).get(k)) for k in COUNTS}
        for arm in ("clinical", "clinical_plus_reading"):
            outcome_rows += metric_rows({"outcome": name, "arm": arm}, result.get("metrics", {}).get(arm, {}), c, minimum, screened)
        pairs += paired_rows({"left": "clinical_plus_reading", "right": "clinical", "scope": name}, result.get("paired_increment", {}), c, minimum, screened)
        associations.append(screen_row({"outcome": name, "scope": "association_not_increment"},
                            {"spearman_abnormal_fraction": result.get("correlation_abnormal_fraction"),
                             "expert_abnormal_fraction_auroc": result.get("expert_abnormal_fraction_auroc")}, c, minimum, screened=screened))
        raw_cells = result.get("reading_outcome_cells", [])
        cells_ok = all(number(cell.get("mothers")) is not None and number(cell["mothers"]) >= minimum for cell in raw_cells)
        for cell in raw_cells:
            if cell.get("reading_positive") not in (0, 1) or cell.get("outcome_positive") not in (0, 1):
                continue
            cc = {"mothers": cell.get("mothers") if cells_ok else None}
            cells.append(screen_row({"outcome": name, "reading_positive": int(cell["reading_positive"]), "outcome_positive": int(cell["outcome_positive"])},
                         {"records": cell.get("records"), "mothers": cell.get("mothers")}, cc, minimum, class_counts=False, screened=screened))
        for cohort_name in ("observed_valid", "missing_or_invalid"):
            co = result.get("cohorts", {}).get(cohort_name, {})
            co_counts = {"mothers": co.get("mothers"), "positive_mothers": co.get("reading_positive_mothers"), "negative_mothers": co.get("reading_negative_mothers")}
            missingness.append(screen_row({"outcome": name, "cohort": cohort_name}, {k: co.get(k) for k in
                                ("records", "mothers", "reading_positive_records", "reading_negative_records", "reading_positive_mothers", "reading_negative_mothers")},
                                co_counts, minimum, screened=screened))
            for feature in ("maternal_age", "gestational_age", "abnormal_fraction", "longest_abnormal_run"):
                descriptor = co.get(feature, {})
                observed_mothers, missing_mothers = number(descriptor.get("observed_mothers")), number(descriptor.get("missing_mothers"))
                descriptors.append(screen_row({"outcome": name, "cohort": cohort_name, "feature": feature},
                                   {k: descriptor.get(k) for k in ("observed_records", "observed_mothers", "missing_records", "missing_mothers", "mean", "median")},
                                   co_counts, minimum, extra_cells=(observed_mothers, minimum if missing_mothers == 0 else missing_mothers), screened=screened))
    tables["outcomes"] = pd.DataFrame(outcome_rows)
    tables["outcome_associations"] = pd.DataFrame(associations)
    tables["outcome_reading_cells"] = pd.DataFrame(cells)
    tables["outcome_missingness"] = pd.DataFrame(missingness)
    tables["outcome_cohort_descriptors"] = pd.DataFrame(descriptors)
    emr = load_json(run, "supplementary/emr_added.json")
    pairs += paired_rows({"left": "cat28_emr", "right": "cat28", "scope": "shared_holdout"}, emr.get("paired_increment", {}), test_counts, minimum, screened)
    if emr.get("metrics"):
        tables["model_comparison"] = pd.concat([tables["model_comparison"], pd.DataFrame(metric_rows(
            {"model": "cat28_emr", "scope": "shared_holdout"}, emr["metrics"], test_counts, minimum, screened))], ignore_index=True)
    rows = []
    for row in emr.get("matched_seed_comparisons", []):
        for pair in paired_rows({"left": "cat28_emr", "right": "cat28"}, row.get("paired_increment", {}), test_counts, minimum, screened):
            pair["seed"] = number(row.get("seed"))
            rows.append(pair)
    tables["emr_matched_seeds"] = pd.DataFrame(rows)
    selection_rows = []
    for row in load_table(run, "supplementary/emr_validation_selection.csv").to_dict("records"):
        if row.get("arm") in ("cat28", "with_emr"):
            selection_rows.append(screen_row({"arm": row["arm"], "selection_stage": "candidate"},
                                  {k: row.get(k) for k in ("seed", "validation_auprc", "threshold")}, counts["val"], minimum, screened=screened))
    for arm in ("cat28", "with_emr"):
        chosen = emr.get("selection", {}).get(arm)
        if isinstance(chosen, dict):
            selection_rows.append(screen_row({"arm": arm, "selection_stage": "selected"},
                                  {k: chosen.get(k) for k in ("seed", "validation_auprc", "threshold")}, counts["val"], minimum, screened=screened))
    tables["emr_selection"] = pd.DataFrame(selection_rows)

    # Site names become local ordinal pseudonyms. No key/mapping is exported.
    loso = load_json(run, "supplementary/loso_metrics.json")
    loso_pred = load_table(run, "supplementary/loso_predictions.csv")
    rows = []
    for index, (site, result) in enumerate(sorted(loso.items()), 1):
        if not isinstance(result, dict):
            continue
        for arm in ("raw", "platt"):
            subset = loso_pred.loc[(loso_pred.site == site) & (loso_pred.arm == arm)] if {"site", "arm"}.issubset(loso_pred) else pd.DataFrame()
            label = {"site": f"S{index:02d}" if screened else site, "arm": arm}
            c = cohort_counts(subset)
            rows += metric_rows(label, result.get(arm, {}), c, minimum, screened)
            if not subset.empty and {"record_all_normal", "record_id", "score", "threshold"}.issubset(subset):
                normal = subset.loc[subset.record_all_normal == 1].assign(alarm=lambda f: f.score >= f.threshold)
                record_alarm = normal.groupby("record_id").alarm.max()
                normal_mothers = normal.mother_id.nunique()
                alarm_mothers = normal.loc[normal.alarm, "mother_id"].nunique()
                quiet_mothers = normal_mothers - alarm_mothers
                allow_alarm = not screened or (adequate(c, minimum) and min(normal_mothers, alarm_mothers, quiet_mothers) >= minimum)
                for key, value in {"normal_records": len(record_alarm), "normal_records_alarmed": int(record_alarm.sum())}.items():
                    rows[-1][key] = value if allow_alarm else None
                if not allow_alarm:
                    for key in ("normal_record_alarm_rate", "false_alarms_per_normal_hour"):
                        for suffix in ("", "_lo", "_hi"):
                            rows[-1][key + suffix] = None
    tables["loso"] = pd.DataFrame(rows)

    official_rows = []
    yolo = load_json(run, "official/yolo_metrics.json")
    yp = load_table(run, "official/yolo_predictions.csv")
    if not yp.empty and "split" not in yp:
        assignment = load_table(run, "splits/segments.csv")
        if {"record_id", "seg_idx", "split"}.issubset(assignment):
            yp = yp.merge(assignment[["record_id", "seg_idx", "split"]], on=["record_id", "seg_idx"], validate="one_to_one")
        else:
            yp = pd.DataFrame()
    if "split" in yp:
        yp = yp.loc[yp.split == "test"]
    yc = cohort_counts(yp)
    for key, model in (("metrics", "official_yolo"), ("cat28_same_image_subset", "cat28_same_image_subset")):
        if key in yolo:
            official_rows += metric_rows({"model": model, "scope": "common_image_subset_training_overlap_unknown"}, yolo[key], yc, minimum, screened)
    ypairs = yolo.get("paired_cat28_minus_yolo", yolo.get("paired_cat28_minus_official_yolo", yolo.get("paired", {})))
    pairs += paired_rows({"left": "cat28_same_image_subset", "right": "official_yolo", "scope": "common_image_subset"}, ypairs, yc, minimum, screened)
    xgb = load_json(run, "official/xgboost_emergency_metrics.json")
    xp = load_table(run, "official/xgboost_emergency_predictions.csv")
    for mode in ("original", "corrected"):
        if mode in xgb:
            sub = xp.loc[xp["parser"] == mode] if "parser" in xp else (xp.loc[xp["mode"] == mode] if "mode" in xp else pd.DataFrame())
            official_rows += metric_rows({"model": "official_xgb_" + mode, "scope": "Emergency_different_task_training_overlap_unknown"}, xgb[mode], cohort_counts(sub), minimum, screened)
    tables["official_reference"] = pd.DataFrame(official_rows)
    tables["paired_comparisons"] = pd.DataFrame(pairs)
    tables["hypothesis_evidence"] = hypothesis_evidence(tables, cfg.get("profile"))
    return tables


def hypothesis_evidence(tables, profile=None):
    rows = []
    pairs = tables.get("paired_comparisons", pd.DataFrame())
    for hypothesis, question, right, caution in (
        ("H1", "인자 측정은 유효하며 판독을 얼마나 예측하는가", None, "품질표·주석 대응과 AP·유병률을 함께 확인; 측정 불가 특성으로 임상 부재를 주장하지 않음"),
        ("H2", "비선형 단일 인자보다 조합이 나은가", "best_single_nonlinear", "검증셋으로 선택한 비선형 단일 인자와 동일 holdout 비교; 인과·상호작용의 증거 아님"),
        ("H3", "추가 인자와 전처리의 기여는 무엇인가", "cat18", "Cat18/Cat28 및 평활 유무·특성군 제거를 짝비교; 부가 대비는 탐색적"),
        ("H4", "작은 CNN과 특성 모델의 성능은 어떤가", "cnn_small_channel_maxabs", "0 포함은 동등성 증명이 아님; 정규화는 같은 width·seed로 별도 비교"),
        ("H5", "판독 연관성과 추가 아웃컴 예측 가치는 다른가", None, "전문의 판독 요약으로 연관성과 증분을 분리; 결측·모델 예측 활용과 구분"),
        ("LOSO", "다른 사이트에서도 경보 기준이 유지되는가", None, "AUROC와 실제 민감도·특이도·정상 기록 경보율을 함께 확인; 내부 사이트 검증"),
    ):
        metric = "auroc" if hypothesis == "H2" else "auprc"
        row = {"question": hypothesis, "research_question": question, "contrast": "cat28_minus_" + right if right else "see_domain_tables",
               "primary_metric": metric if right else "not_applicable", "difference": None, "ci95_low": None, "ci95_high": None,
               "evidence": "see_domain_tables", "interpretation_limit": caution}
        if right and {"left", "right", "metric", "scope"}.issubset(pairs):
            selected = pairs.loc[(pairs.left == "cat28") & (pairs.right == right) & (pairs.metric == metric) & (pairs.scope == "shared_holdout")]
            if not selected.empty:
                value = selected.iloc[0]
                row.update({k: number(value.get(k)) for k in ("difference", "ci95_low", "ci95_high")})
                lo, hi = row["ci95_low"], row["ci95_high"]
                row["evidence"] = "not_available_or_suppressed" if lo is None or hi is None else (
                    "positive_interval" if lo > 0 else "negative_interval" if hi < 0 else "inconclusive_not_equivalent")
        if profile == "mock":
            row["evidence"] = "execution_test_only"
        rows.append(row)
    return pd.DataFrame(rows)


def write_review_report(out, tables, profile, image_files=()):
    """No arbitrary HTML/links/labels from the internal report are accepted."""
    title = "MOCK — 반출 심사용 집계 결과 (학술 결론 금지)" if profile == "mock" else "반출 심사용 집계 결과 — 기관 승인 전"
    notes = ["이 묶음은 반출 승인이 아닙니다. 수신 기관의 검토가 완료될 때까지 현장에 보관하세요.",
             "기존 집계 CSV는 csv/, 추가 데이터·학습·방문 진단 CSV는 visit_audit/csv/에 있습니다. 이미지 후보는 images/와 visit_audit/images/입니다. onsite_figures/는 개별 SHAP·원 기관 코드가 있는 현장용입니다.",
             "작거나 확인되지 않은 산모 수/양성·음성 산모 수의 집계는 억제됩니다. 기준을 통과해도 기관의 반출 기준 충족을 보장하지 않습니다.",
             "표의 공란은 미산출 또는 억제이며 0이 아닙니다. 작은 셀은 다른 표와의 차분으로 추론될 수도 있어 기관 검토가 필요합니다.",
             "산모 군집 부트스트랩 신뢰구간은 고정된 모델의 표본 불확실성입니다. 0을 포함하는 차이는 동등성의 증명이 아닙니다.",
             "특성 기여·관측 연관성은 인과적 원인이 아닙니다. STV 대용치는 샘플 간 차분이며 임상 beat-to-beat STV가 아닙니다.",
             "LOSO는 내부 사이트 제외 평가입니다. 확률 보정만으로 경보 부담이 개선되었다고 해석하지 않습니다.",
             "추론 비용은 측정 범위별 값입니다. 인자 추출 시간은 별도 측정한 평균이며, 예측 시간과 더한 값을 실측 종단 간 지연으로 주장하지 않습니다.",
             "공식 모델은 별도 과제/이미지 부분집합 참고입니다. 배포 가중치의 원학습 중복 여부는 미확인입니다."]
    markup = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>Aggregate review report</title>",
              "<style>body{font:15px system-ui;max-width:1400px;margin:2em auto;padding:1em}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}.scroll{overflow:auto}img{max-width:100%}</style>",
              "<h1>" + title + "</h1>"]
    markdown = ["# " + title]
    for note in notes:
        markup.append("<p>" + html.escape(note) + "</p>")
        markdown.append(note)
    if image_files:
        intro = "이미지 전용 심사는 images/의 PNG만 선택합니다. CSV·HTML·JSON·PDF는 이 폴더 밖의 현장 검토 자료입니다. 그래프 이미지도 기관의 승인이 필요합니다."
        markup.append("<h2>PNG 그래프 묶음</h2><p>" + intro + "</p>")
        markdown += ["## PNG 그래프 묶음", intro]
        for path in image_files:
            relative = path.relative_to(out).as_posix()
            markup.append("<details><summary>" + path.name + "</summary><img loading='lazy' src='" + relative + "' alt='" + path.stem + "'></details>")
            markdown.append("![" + path.stem + "](" + relative + ")")
    figure_files = []
    comparison = tables.get("model_comparison", pd.DataFrame())
    if not comparison.empty and "auroc" in comparison and comparison.auroc.notna().any():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, len(comparison) * .27)))
        for axis, key in zip(axes, ("auroc", "auprc")):
            selected = comparison.loc[comparison[key].notna()].sort_values(key)
            axis.barh(selected.model, selected[key], color="#327A9B")
            axis.hlines(np.arange(len(selected)), selected[key + "_lo"], selected[key + "_hi"], color="black")
            axis.set(xlim=(0, 1), xlabel=key.upper(), title="Shared holdout / cluster CI")
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        for suffix in ("png", "pdf"):
            path = out / "figures" / ("model_comparison." + suffix)
            fig.savefig(path, bbox_inches="tight")
            figure_files.append(path)
        plt.close(fig)
        markup.append("<img src='figures/model_comparison.png' alt='Model comparison'>")
        markdown.append("![Model comparison](figures/model_comparison.png)")
    response = tables.get("feature_response_train", pd.DataFrame())
    response_features = [] if response.empty else [feature for feature in ("stv", "fhr_min", "figo_baseline_var", "n_decel")
                         if response.loc[response.feature == feature, "positive_fraction"].notna().any()]
    if response_features:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(response_features), figsize=(4 * len(response_features), 3.5), squeeze=False)
        for axis, feature in zip(axes[0], response_features):
            selected = response.loc[(response.feature == feature) & response.positive_fraction.notna()].sort_values("bin")
            axis.plot(selected.bin, selected.positive_fraction, "o-", color="#327A9B")
            axis.set(title=feature, xlabel="Training quantile bin", ylabel="Abnormal fraction", ylim=(0, 1))
        fig.suptitle("Screened training aggregates; marginal association, not causality")
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        for suffix in ("png", "pdf"):
            path = out / "figures" / ("feature_response_training." + suffix)
            fig.savefig(path, bbox_inches="tight")
            figure_files.append(path)
        plt.close(fig)
        markup.append("<img src='figures/feature_response_training.png' alt='Training feature response'>")
        markdown.append("![Training feature response](figures/feature_response_training.png)")
    for name, frame in tables.items():
        # Names are fixed in this module; DataFrame strings are allowlisted above.
        markup.append("<h2>" + name.replace("_", " ") + "</h2><div class='scroll'>")
        markup.append(f"<p><a href='csv/{name}.csv'>CSV</a></p>")
        markup.append(frame.to_html(index=False, na_rep="suppressed / unavailable", float_format=lambda v: f"{v:.5g}") if len(frame.columns) else "<p>Not available</p>")
        markup.append("</div>")
        markdown.append("## " + name.replace("_", " "))
        markdown.append(f"[CSV](csv/{name}.csv)")
        markdown.append(frame.to_markdown(index=False, floatfmt=".5g") if len(frame.columns) else "Not available")
    markup.append("</html>")
    markup.insert(-1, "<p><a href='visit_audit/index.html'>반출 검토용 추가 집계 · 학습곡선 · 방문 진단</a> (실행 일지 종료 시 생성)</p>")
    (out / "report.html").write_text("\n".join(markup), encoding="utf-8")
    (out / "report.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    return [out / "report.html", out / "report.md"] + figure_files


def _build_export_review(run, out, cfg):
    out = Path(out)
    if out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError("Symbolic-link export destinations are not accepted")
    out.mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(exist_ok=True)
    tables = collect_tables(run, cfg)
    files = []
    for name, frame in tables.items():
        path = out / "csv" / (name + ".csv")
        # Preserve a readable schema even for absent optional analyses.
        (frame if len(frame.columns) else pd.DataFrame(columns=["review_status"])).to_csv(path, index=False)
        files.append(path)
    profile = cfg.get("profile") if cfg.get("profile") in ("mock", "full") else "unspecified"
    metadata = {"profile": profile, "review_status": "pending_institution_review",
                "export_min_mothers": max(1, int(cfg.get("export_min_mothers", 10))),
                "screening_is_approval": False, "seed": number(cfg.get("seed")), "design_version": "2.0",
                "primary_metric": "auprc", "H2_primary_metric": "auroc", "H2_secondary_metric": "auprc",
                "H4_equivalence_or_noninferiority_claim": False, "split_and_bootstrap_unit": "mother"}
    history = load_json(run, "splits/cohort_history.json")
    metadata["cohort_history"] = history.get("status") if history.get("status") in ("checked", "not_checked") else "not_checked"
    # History counts are useful provenance but still screened for small cohorts.
    for key in ("supplied_mothers", "matched_mothers", "matched_validation_mothers", "matched_test_mothers"):
        value = number(history.get(key))
        metadata["cohort_history_" + key] = value if value is not None and (value == 0 or value >= metadata["export_min_mothers"]) else None
    budget = cfg.get("budget", {})
    for key in ("cv_folds", "tree_iterations", "tree_patience", "cnn_epochs", "cnn_patience", "bootstrap", "batch_size"):
        if number(budget.get(key)) is not None:
            metadata[key] = number(budget[key])
    metadata["seeds"] = [number(v) for v in budget.get("seeds", []) if number(v) is not None]
    manifest = load_json(run, "run_manifest.json")
    package_hash = manifest.get("package_sha256")
    if isinstance(package_hash, str) and re.fullmatch(r"[0-9a-fA-F]{64}", package_hash):
        metadata["package_sha256"] = package_hash.lower()
    protocol = out / "protocol_summary.json"
    write_json(protocol, metadata)
    files.append(protocol)
    from .export_images import write_image_atlas
    image_files = write_image_atlas(out / "images", tables, metadata)
    files += image_files
    files += write_review_report(out, tables, profile, image_files=image_files)
    hashes = [{"file": path.relative_to(out).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "bytes": path.stat().st_size} for path in sorted(files)]
    write_json(out / "EXPORT_MANIFEST.json", {"schema_version": 3, "review_status": "pending_institution_review",
               "manifest_scope": "screened_aggregates_only", "csv_directory": "csv",
               "onsite_directory_prefix": "onsite_figures", "whole_directory_exportable": False,
               "image_submission_directory": "images", "image_submission_format": "RGB PNG only",
               "screening_is_approval": False, "source_material_included": False,
               "excluded_categories": ["identifiers", "individual_predictions", "raw_signals", "model_weights", "source_paths", "source_file_hashes"],
               "files": hashes})
    validate_review_bundle(out)


def run_export_review(run, out, cfg):
    """Build privately, then publish atomically; interrupted builds never leak partial files."""
    out = Path(out)
    if out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError("Symbolic-link export destinations are not accepted")
    diagnostics = []
    if out.exists() and any(out.iterdir()):
        # A crash after atomic rename but before the stage marker is recoverable.
        if (out / "EXPORT_MANIFEST.json").is_file():
            validate_review_bundle(out)
            return
        # A diagnostic-only visit may precede the first completed analysis.
        from .export_diagnostics import DIRECTORY, validate_diagnostics
        for directory in out.iterdir():
            if not directory.is_dir() or not DIRECTORY.fullmatch(directory.name):
                raise ValueError("Incomplete export bundle contains unexpected files")
            validate_diagnostics(directory)
            diagnostics.append(directory)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-review-build-", dir=Path(run)) as temporary:
        staged = Path(temporary) / "bundle"
        _build_export_review(run, staged, cfg)
        for directory in diagnostics:
            shutil.copytree(directory, staged / directory.name)
        validate_review_bundle(staged)
        backup = None
        if out.exists():
            if diagnostics:
                backup = Path(temporary) / "previous_diagnostics"
                out.replace(backup)
            else:
                out.rmdir()  # Only the pre-created EMPTY stage directory can be removed.
        try:
            staged.replace(out)
        except BaseException:
            if backup is not None:
                backup.replace(out)
            raise


EXPORT_TABLES = {"model_comparison", "paired_comparisons", "cohort_counts", "cv_summary", "validation_selection",
                 "single_feature_selection", "seed_holdout_metrics", "seed_paired_cat28_minus_comparator", "feature_importance",
                 "feature_response_train", "feature_quality", "cnn_capacity_validation", "cnn_selection", "cnn_capacity_summary",
                 "normalization_matched", "inference_cost", "outcomes", "outcome_associations", "outcome_reading_cells",
                 "outcome_missingness", "outcome_cohort_descriptors", "emr_matched_seeds", "emr_selection", "loso", "official_reference", "hypothesis_evidence", "preprocessing_sensitivity", "figo_agreement"}


def validate_review_bundle(out):
    """Verify screened aggregates and separately validate any onsite-only guides."""
    from html.parser import HTMLParser
    root = Path(out)
    if root.is_symlink():
        raise ValueError("Symbolic-link review bundle is not accepted")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Symbolic links are not allowed in a review bundle")
    actual = {path.relative_to(root).as_posix() for path in paths if path.is_file()}
    manifest = read_json(root / "EXPORT_MANIFEST.json")
    schema = manifest.get("schema_version", 1)
    onsite = {}
    diagnostics = {}
    from .export_diagnostics import DIRECTORY, validate_diagnostics
    for directory in root.iterdir():
        if directory.is_dir() and DIRECTORY.fullmatch(directory.name):
            diagnostics[directory.name] = validate_diagnostics(directory)
    # Additive screened supplements have their own immutable inventories.
    # This also supports adding diagnostics to a completed schema 1/2 run.
    actual = {name for name in actual if name.split("/")[0] not in diagnostics}
    if schema == 3:
        if (manifest.get("manifest_scope") != "screened_aggregates_only"
                or manifest.get("csv_directory") != "csv"
                or manifest.get("onsite_directory_prefix") != "onsite_figures"
                or manifest.get("whole_directory_exportable") is not False):
            raise ValueError("Invalid review workspace scope")
        from .onsite_figures import ONSITE_DIRECTORY, validate_onsite_figures
        for directory in root.iterdir():
            if directory.is_dir() and ONSITE_DIRECTORY.fullmatch(directory.name):
                onsite[directory.name] = validate_onsite_figures(directory)
        # These files have their own inventory/hashes and are not screened exports.
        actual = {name for name in actual if name.split("/")[0] not in onsite}
    elif schema not in (1, 2):
        raise ValueError("Unsupported review schema")
    csv_prefix = "csv/" if schema == 3 else ""
    allowed = {csv_prefix + name + ".csv" for name in EXPORT_TABLES} | {
        "report.html", "report.md", "protocol_summary.json", "EXPORT_MANIFEST.json", "figures/model_comparison.png", "figures/model_comparison.pdf",
        "figures/feature_response_training.png", "figures/feature_response_training.pdf"}
    from .export_images import IMAGE_NAME, validate_image_directory
    allowed |= {name for name in actual if name.startswith("images/") and IMAGE_NAME.fullmatch(name.removeprefix("images/"))}
    if actual - allowed:
        raise ValueError("Unrecognized files in aggregate review bundle")
    if manifest.get("review_status") != "pending_institution_review" or manifest.get("screening_is_approval") is not False:
        raise ValueError("Review status missing or invalid")
    if schema >= 2 or (root / "images").exists():
        validate_image_directory(root / "images")
    listed = {entry["file"] for entry in manifest["files"]}
    if listed != actual - {"EXPORT_MANIFEST.json"} or len(listed) != len(manifest["files"]):
        raise ValueError("Review manifest inventory mismatch")
    for entry in manifest["files"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            raise ValueError("Review manifest checksum mismatch")

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in {"script", "iframe", "object", "embed", "form"}:
                raise ValueError("Active content is not allowed in a review report")
            for key, value in attrs:
                allowed_images = {name for name in actual if name.startswith("images/") and IMAGE_NAME.fullmatch(name.removeprefix("images/"))}
                allowed_images |= {"figures/model_comparison.png", "figures/feature_response_training.png"}
                allowed_links = allowed_images | {name for name in actual if name.startswith("csv/") and name.endswith(".csv")} | {"visit_audit/index.html"}
                if key.startswith("on") or (key == "src" and value not in allowed_images) or (key == "href" and value not in allowed_links):
                    raise ValueError("Unexpected link in aggregate review report")

    Links().feed((root / "report.html").read_text(encoding="utf-8"))
    return {"status": "validated_pending_institution_review", "files": len(actual), "manifest_hashes_match": True,
            "onsite_only": onsite, "screened_diagnostics": diagnostics, "whole_directory_exportable": False}
