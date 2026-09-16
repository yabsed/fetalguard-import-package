"""Report §5.3: outcomes, annotation agreement, sites, added metadata."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import note, table, write_json
from .data import EMR_FEATURES
from .features import CAT28
from .evaluation import group_folds, threshold90, evaluate, paired
from .models import with_metadata, fit_tree, predict


def calibration_bins(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for k in range(bins):
        mask = (p >= k / bins) & ((p < (k + 1) / bins) if k < bins - 1 else (p <= 1))
        if mask.any():
            rows.append(dict(bin=k, n=int(mask.sum()), mean_prediction=float(p[mask].mean()), observed=float(y[mask].mean())))
    return rows


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def outcome_cohort_summary(frame):
    """Record-level aggregate denominators; do not substitute segment counts."""
    summary = dict(records=len(frame), mothers=int(frame.mother_id.nunique()),
                   reading_positive_records=int(frame.any_abnormal.eq(1).sum()),
                   reading_negative_records=int(frame.any_abnormal.eq(0).sum()),
                   reading_positive_mothers=int(frame.loc[frame.any_abnormal.eq(1), "mother_id"].nunique()),
                   reading_negative_mothers=int(frame.loc[frame.any_abnormal.eq(0), "mother_id"].nunique()))
    for column in ("maternal_age", "gestational_age", "abnormal_fraction", "longest_abnormal_run"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            summary[column] = dict(observed_records=int(values.notna().sum()),
                                   observed_mothers=int(frame.loc[values.notna(), "mother_id"].nunique()),
                                   missing_records=int(values.isna().sum()),
                                   missing_mothers=int(frame.loc[values.isna(), "mother_id"].nunique()),
                                   mean=float(values.mean()) if values.notna().any() else None,
                                   median=float(values.median()) if values.notna().any() else None)
    summary["clinical_field_completeness"] = {
        column: dict(observed_records=int(frame[column].notna().sum()),
                     observed_mothers=int(frame.loc[frame[column].notna(), "mother_id"].nunique()),
                     missing_records=int(frame[column].isna().sum()),
                     missing_mothers=int(frame.loc[frame[column].isna(), "mother_id"].nunique()))
        for column in EMR_FEATURES if column in frame}
    return summary


def outcome_association(valid, column):
    cells = []
    for reading in (0, 1):
        for outcome in (0, 1):
            sub = valid[valid.any_abnormal.eq(reading) & valid.target.eq(outcome)]
            cells.append(dict(reading_positive=reading, outcome_positive=outcome,
                              records=len(sub), mothers=int(sub.mother_id.nunique())))
    correlation = (float(spearmanr(valid[column], valid.abnormal_fraction).statistic)
                   if valid[column].nunique() > 1 and valid.abnormal_fraction.nunique() > 1 else None)
    return dict(correlation_abnormal_fraction=correlation,
                reading_presence_cross_table=pd.crosstab(valid.any_abnormal, valid.target).to_dict(),
                reading_outcome_cells=cells,
                expert_abnormal_fraction_auroc=(float(roc_auc_score(valid.target, valid.abnormal_fraction))
                                                if valid.target.nunique() == 2 else None))


def outcomes(run, out, cfg):
    records = table(run / "data/records.csv")
    for c in EMR_FEATURES:
        if c not in records:
            records[c] = np.nan
    labels = {"ph_lt_7_20": ("emr_UA.pH", 7.2, 6, 8),
              "apgar1_lt_7": ("emr_APGAR.1min", 7, 0, 10), "apgar5_lt_7": ("emr_APGAR.5min", 7, 0, 10)}
    summaries, predictions = {}, []
    for name, (column, cutoff, low, high) in labels.items():
        if column not in records:
            summaries[name] = {"status": "missing_field", "field": column,
                               "cohorts": {"observed_valid": outcome_cohort_summary(records.iloc[:0]),
                                           "missing_or_invalid": outcome_cohort_summary(records)}}
            continue
        observed = records[column].between(low, high)
        valid = records[observed].copy().reset_index(drop=True)
        valid["target"] = (valid[column] < cutoff).astype(int)
        description = dict(field=column, cutoff=cutoff, valid_range=[low, high], n=len(valid),
                           mothers=int(valid.mother_id.nunique()), positives=int(valid.target.sum()),
                           positive_mothers=int(valid.loc[valid.target.eq(1), "mother_id"].nunique()),
                           negative_mothers=int(valid.loc[valid.target.eq(0), "mother_id"].nunique()),
                           cohorts={"observed_valid": outcome_cohort_summary(valid),
                                    "missing_or_invalid": outcome_cohort_summary(records.loc[~observed])},
                           reading_source="expert-provided abnormality labels, not model predictions",
                           reading_features=["abnormal_fraction", "longest_abnormal_run"],
                           reading_note="Longest run is missing when recording continuity cannot be established. CatBoost handles this missingness; reading association and added predictive value answer separate questions.",
                           analysis_scope="exploratory, mother-grouped record-level OOF over records with observed outcome; not the sealed segment holdout")
        description.update(outcome_association(valid, column))
        class_mothers = valid.groupby("target").mother_id.nunique()
        k = cfg["budget"]["cv_folds"]
        if len(class_mothers) < 2 or class_mothers.min() < max(6, k * 2):
            summaries[name] = dict(description, status="insufficient_outcome_groups", min_groups=max(6, k * 2))
            continue
        try:
            pair_rows = []
            for fold, (tr, te) in enumerate(group_folds(valid, k, cfg["seed"])):
                pool = valid.iloc[tr].reset_index(drop=True)
                it, iv = next(group_folds(pool, 3, cfg["seed"] + fold))
                for arm, cols in {"clinical": EMR_FEATURES, "clinical_plus_reading": EMR_FEATURES + ["abnormal_fraction", "longest_abnormal_run"]}.items():
                    note(f"H5 {name} fold={fold + 1} {arm}")
                    model = fit_tree(pool.iloc[it], pool.iloc[iv], cols, cfg, cfg["seed"], path=out / "models" / f"{name}_{fold}_{arm}.cbm")
                    vp, p = predict(model, pool.iloc[iv], cols), predict(model, valid.iloc[te], cols)
                    threshold = threshold90(pool.iloc[iv].target, vp)
                    row = valid.iloc[te][["record_id", "mother_id", "target"]].assign(outcome=name, arm=arm, fold=fold, score=p, threshold=threshold)
                    pair_rows.append(row)
            pred = pd.concat(pair_rows, ignore_index=True)
            predictions.append(pred)
            left, right = [pred[pred.arm == a] for a in ("clinical_plus_reading", "clinical")]
            join = left.merge(right[["record_id", "score"]], on="record_id", suffixes=("", "_baseline"), validate="one_to_one")
            summaries[name] = dict(description, status="ok",
                metrics={a: evaluate(sub, sub.score, sub.threshold, cfg["budget"]["bootstrap"], cfg["seed"])
                         for a, sub in pred.groupby("arm")},
                paired_increment=paired(join, join.score, join.score_baseline, cfg["budget"]["bootstrap"], cfg["seed"]))
        except ValueError as exc:
            if "두 클래스" not in str(exc):
                raise
            summaries[name] = dict(description, status="insufficient_classes_in_grouped_fold", reason=str(exc))
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "outcome_oof_predictions.csv", index=False)
    write_json(out / "outcomes.json", summaries)


def annotations(run, out):
    records, segments = table(run / "data/records.csv"), table(run / "data/segments.csv")
    mapping = [("BaseLine", "figo_baseline", "continuous"), ("Baseline_Variability", "figo_baseline_var", "ordinal"),
               ("Acceleration", "n_accel", "acceleration"), ("Early_deceleration", "n_early_decel", "binary"),
               ("Late_deceleration", "n_late_decel", "binary"), ("Variable_deceleration", "n_variable_decel", "binary"),
               ("Prolonged_deceleration", "n_prolonged_decel", "binary")]
    rows, details = [], []
    for label, feature, kind in mapping:
        target = "label_" + label
        if target not in records:
            rows.append(dict(label=label, feature=feature, status="missing_field"))
            continue
        agg = segments.groupby("record_id")[feature].agg("mean" if kind in {"continuous", "ordinal"} else "sum")
        joined = records[["record_id", target]].join(agg, on="record_id").dropna()
        y, x = joined[target].to_numpy(), joined[feature].to_numpy()
        if len(joined) < 5 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
            rows.append(dict(label=label, feature=feature, status="constant_or_insufficient", n=len(joined)))
            continue
        row = dict(label=label, feature=feature, status="ok", n=len(joined), spearman=float(spearmanr(x, y).statistic))
        if kind == "continuous":
            row["mae"] = float(np.mean(np.abs(x - y)))
        if kind in {"acceleration", "binary"}:
            y = (y == 1).astype(int) if kind == "acceleration" else (y > 0).astype(int)
            row["auroc"] = float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 else np.nan
        rows.append(row)
        details.append(joined.rename(columns={target: "label_value", feature: "factor_value"}).assign(label=label, factor=feature))
    pd.DataFrame(rows).to_csv(out / "figo_annotation_agreement.csv", index=False)
    if details:
        pd.concat(details).to_csv(out / "figo_record_pairs.csv", index=False)
    write_json(out / "annotation_scope.json", {"unit": "record", "note": "Record labels are compared to mean/sum of selected fetus windows. No claim of segment-level gold standard or inter-rater reliability. annotation_person placeholder distributions are in data/annotation_fields.csv."})


def sites(run, out, cfg):
    frame = with_metadata(run)
    results, predictions, bins = {}, [], []
    for site in sorted(frame.site.unique()):
        hold = frame[frame.site == site].copy()
        # Exclude the held site's mothers everywhere, including cross-site repeats.
        development = frame[~frame.mother_id.isin(hold.mother_id)].reset_index(drop=True)
        if hold.target.nunique() < 2 or development.target.nunique() < 2 or development.mother_id.nunique() < 8:
            results[site] = {"status": "insufficient_classes_or_groups"}
            continue
        tr, va = next(group_folds(development, 5, cfg["seed"]))
        train, val = development.iloc[tr], development.iloc[va]
        if train.target.nunique() < 2 or val.target.nunique() < 2:
            results[site] = {"status": "insufficient_validation_classes"}
            continue
        note(f"LOSO held site={site}: train={len(train):,}, test={len(hold):,}")
        model = fit_tree(train, val, CAT28, cfg, cfg["seed"], path=out / "models" / f"loso_{site}.cbm")
        vp, hp = predict(model, val, CAT28), predict(model, hold, CAT28)
        calibrator = LogisticRegression(random_state=cfg["seed"]).fit(logit(vp), val.target)
        cp = calibrator.predict_proba(logit(hp))[:, 1]
        cvp = calibrator.predict_proba(logit(vp))[:, 1]
        results[site] = {"status": "ok", "held_mothers": int(hold.mother_id.nunique()),
                         "source_only_calibration": {"coef": calibrator.coef_.tolist(), "intercept": calibrator.intercept_.tolist(),
                                                     "monotone_increasing": bool(calibrator.coef_[0, 0] > 0)},
                         "operating_rule": "threshold chosen for at least 90% specificity on source validation only, separately for raw and Platt scores",
                         "calibration_interpretation": "An increasing Platt map preserves ranking; reselecting the same source-specificity threshold usually preserves alarms. Calibration evaluates probability accuracy, not a solution to site alarm shift.",
                         "validation_scope": "held source-site code; no claim of independent prospective external validation"}
        operating_predictions = {}
        for arm, p, v in (("raw", hp, vp), ("platt", cp, cvp)):
            threshold = threshold90(val.target, v)
            operating_predictions[arm] = p >= threshold
            results[site][arm] = evaluate(hold, p, threshold, cfg["budget"]["bootstrap"], cfg["seed"])
            results[site][arm]["threshold"] = threshold
            predictions.append(hold[["record_id", "mother_id", "seg_idx", "target", "record_all_normal"]].assign(site=site, arm=arm, score=p, threshold=threshold))
            bins.extend([dict(site=site, arm=arm, **row) for row in calibration_bins(hold.target, p)])
        results[site]["raw_platt_same_operating_predictions"] = bool(np.array_equal(operating_predictions["raw"], operating_predictions["platt"]))
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "loso_predictions.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "loso_calibration_bins.csv", index=False)
    write_json(out / "loso_metrics.json", results)


def metadata(run, out, cfg):
    frame = with_metadata(run)
    train, val, test = [frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    opts = cfg["budget"]
    trials, selection, seed_predictions = {}, [], []
    arms = {"cat28": CAT28, "with_emr": CAT28 + EMR_FEATURES}
    for arm, columns in arms.items():
        trials[arm] = []
        for seed in opts["seeds"]:
            path = (run / "experiment_a/models" / f"holdout_cat28_{seed}.cbm" if arm == "cat28"
                    else out / "models" / f"emr_added_{seed}.cbm")
            model = fit_tree(train, val, columns, cfg, seed, path=path)
            vp, tp = predict(model, val, columns), predict(model, test, columns)
            validation_ap = float(average_precision_score(val.target, vp))
            threshold = threshold90(val.target, vp)
            trial = dict(seed=seed, validation_auprc=validation_ap, threshold=threshold,
                         score=tp, model_file=str(path.relative_to(run)))
            trials[arm].append(trial)
            selection.append(dict(arm=arm, seed=seed, validation_auprc=validation_ap, threshold=threshold))
            seed_predictions.append(test[["record_id", "mother_id", "seg_idx", "target", "site", "record_all_normal"]]
                .assign(arm=arm, seed=seed, score=tp, threshold=threshold))
    # Same rule, training split, early stopping and search budget for both arms.
    chosen = {arm: max(candidates, key=lambda row: row["validation_auprc"]) for arm, candidates in trials.items()}
    baseline, added = chosen["cat28"], chosen["with_emr"]
    threshold = added["threshold"]
    joined = test.copy()
    joined["score"], joined["threshold"] = baseline["score"], baseline["threshold"]
    joined["score_emr"] = added["score"]
    joined[["record_id", "mother_id", "seg_idx", "target", "score", "score_emr"]].to_csv(out / "emr_predictions.csv", index=False)
    pd.concat(seed_predictions, ignore_index=True).to_csv(out / "emr_seed_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(out / "emr_validation_selection.csv", index=False)
    seed_comparisons = []
    by_seed = {arm: {row["seed"]: row for row in candidates} for arm, candidates in trials.items()}
    for seed in opts["seeds"]:
        left, right = by_seed["with_emr"][seed], by_seed["cat28"][seed]
        seed_comparisons.append(dict(seed=seed,
            baseline_validation_auprc=right["validation_auprc"], added_validation_auprc=left["validation_auprc"],
            paired_increment=paired(test, left["score"], right["score"], opts["bootstrap"], cfg["seed"])))
    result = dict(features=arms["with_emr"], threshold=threshold,
                  metrics=evaluate(test, added["score"], threshold, cfg["budget"]["bootstrap"], cfg["seed"]),
                  baseline_metrics=evaluate(test, baseline["score"], baseline["threshold"], cfg["budget"]["bootstrap"], cfg["seed"]),
                  paired_increment=paired(joined, joined.score_emr, joined.score, cfg["budget"]["bootstrap"], cfg["seed"]),
                  selection={arm: {key: value for key, value in row.items() if key != "score"} for arm, row in chosen.items()},
                  selection_policy="maximum validation AP over identical configured seed budgets; ties follow configured seed order",
                  configured_seeds=opts["seeds"], matched_seed_comparisons=seed_comparisons,
                  comparison_note="Primary comparison applies the same selection policy to both arms; matched-seed comparisons isolate feature inclusion at each seed. Test-cohort intervals do not include model-selection uncertainty.",
                  note="GA is gestational age at delivery in supplied EMR; availability at monitoring must be established before deployment. No postnatal outcome/Delivery/Emergency enters this model.")
    subgroups = {"age_lt35": joined.maternal_age < 35, "age_ge35": joined.maternal_age >= 35,
                 "age_missing": joined.maternal_age.isna(), "GA_lt37": joined.gestational_age < 37,
                 "GA_ge37": joined.gestational_age >= 37, "GA_missing": joined.gestational_age.isna(),
                 "signal_missing_0": joined.missing_fraction == 0, "signal_missing_positive": joined.missing_fraction > 0}
    subgroup_results = {}
    for name, mask in subgroups.items():
        sub = joined.loc[mask]
        if not len(sub):
            subgroup_results[name] = {"status": "empty"}
            continue
        subgroup_results[name] = {"cat28": evaluate(sub, sub.score, sub.threshold, cfg["budget"]["bootstrap"], cfg["seed"]),
                                  "with_emr": evaluate(sub, sub.score_emr, threshold, cfg["budget"]["bootstrap"], cfg["seed"])}
    write_json(out / "emr_added.json", result)
    write_json(out / "subgroups.json", subgroup_results)
    site_results = {}
    for name, sub in joined.groupby("site"):
        site_results[name] = evaluate(sub, sub.score, sub.threshold, cfg["budget"]["bootstrap"], cfg["seed"])
    write_json(out / "internal_site_metrics.json", site_results)


def run_supplementary(run, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    annotations(run, out)
    outcomes(run, out, cfg)
    sites(run, out, cfg)
    metadata(run, out, cfg)
