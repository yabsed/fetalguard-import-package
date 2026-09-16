"""Report §5.3: outcomes, annotation agreement, sites, added metadata."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
            summaries[name] = {"status": "missing_field", "field": column}
            continue
        valid = records[records[column].between(low, high)].copy().reset_index(drop=True)
        valid["target"] = (valid[column] < cutoff).astype(int)
        class_mothers = valid.groupby("target").mother_id.nunique()
        k = cfg["budget"]["cv_folds"]
        if len(class_mothers) < 2 or class_mothers.min() < max(6, k * 2):
            summaries[name] = dict(status="insufficient_outcome_groups", n=len(valid), positives=int(valid.target.sum()), min_groups=max(6, k * 2))
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
            summaries[name] = dict(status="ok", n=len(valid), positives=int(valid.target.sum()),
                correlation_abnormal_fraction=float(spearmanr(valid[column], valid.abnormal_fraction).statistic),
                reading_presence_cross_table=pd.crosstab(valid.any_abnormal, valid.target).to_dict(),
                metrics={a: evaluate(sub, sub.score, sub.threshold, cfg["budget"]["bootstrap"], cfg["seed"])
                         for a, sub in pred.groupby("arm")},
                paired_increment=paired(join, join.score, join.score_baseline, cfg["budget"]["bootstrap"], cfg["seed"]))
        except ValueError as exc:
            if "두 클래스" not in str(exc):
                raise
            summaries[name] = dict(status="insufficient_classes_in_grouped_fold", reason=str(exc))
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
                         "source_only_calibration": {"coef": calibrator.coef_.tolist(), "intercept": calibrator.intercept_.tolist()}}
        for arm, p, v in (("raw", hp, vp), ("platt", cp, cvp)):
            threshold = threshold90(val.target, v)
            results[site][arm] = evaluate(hold, p, threshold, cfg["budget"]["bootstrap"], cfg["seed"])
            results[site][arm]["threshold"] = threshold
            predictions.append(hold[["record_id", "mother_id", "seg_idx", "target", "record_all_normal"]].assign(site=site, arm=arm, score=p, threshold=threshold))
            bins.extend([dict(site=site, arm=arm, **row) for row in calibration_bins(hold.target, p)])
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out / "loso_predictions.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "loso_calibration_bins.csv", index=False)
    write_json(out / "loso_metrics.json", results)


def metadata(run, out, cfg):
    frame = with_metadata(run)
    train, val, test = [frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    cols = CAT28 + EMR_FEATURES
    model = fit_tree(train, val, cols, cfg, cfg["seed"], path=out / "models/emr_added.cbm")
    vp, tp = predict(model, val, cols), predict(model, test, cols)
    threshold = threshold90(val.target, vp)
    base = table(run / "experiment_a/test_predictions.csv").query("model == 'cat28'")
    joined = test.merge(base[["record_id", "seg_idx", "score", "threshold"]], on=["record_id", "seg_idx"], validate="one_to_one", sort=False)
    joined["score_emr"] = tp
    joined[["record_id", "mother_id", "seg_idx", "target", "score", "score_emr"]].to_csv(out / "emr_predictions.csv", index=False)
    result = dict(features=cols, threshold=threshold, metrics=evaluate(test, tp, threshold, cfg["budget"]["bootstrap"], cfg["seed"]),
                  paired_increment=paired(joined, joined.score_emr, joined.score, cfg["budget"]["bootstrap"], cfg["seed"]),
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
