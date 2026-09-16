"""Validation-only operating points and paired mother-cluster intervals."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from .common import require_two_classes, write_json, table


def group_folds(frame, n_splits, seed, target="target"):
    if frame.mother_id.nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} mothers for grouped splitting")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(frame, frame[target], groups=frame.mother_id):
        if set(frame.iloc[tr].mother_id) & set(frame.iloc[te].mother_id):
            raise AssertionError("Mother leakage")
        yield tr, te


def create_splits(data_dir, out, cfg):
    records = table(data_dir / "records.csv")
    usable = records[records.n_kept > 0].reset_index(drop=True)
    prior_file = cfg.get("prior_cohort_file")
    known = set()
    if prior_file:
        prior = pd.read_csv(Path(prior_file), dtype=str, keep_default_na=False)
        if "mother_id" not in prior or prior.mother_id.str.strip().isin(["", "nan", "None", "9999"]).any():
            raise ValueError("Prior cohort CSV needs nonmissing mother_id; known mothers go to training only")
        known = set(prior.mother_id.str.strip())
    usable["prior_seen"] = usable.mother_id.isin(known)
    eligible = usable.index[~usable.prior_seen].to_numpy()
    fresh = usable.loc[eligible].reset_index(drop=True)
    if fresh.mother_id.nunique() < max(cfg["test_folds"], cfg["validation_folds"] + 1):
        raise ValueError("Too few previously unseen mothers for validation/test; prior cohort cannot form a new confirmation cohort")
    dev, test = next(group_folds(fresh, cfg["test_folds"], cfg["seed"], "any_abnormal"))
    train, val = next(group_folds(fresh.iloc[dev].reset_index(drop=True), cfg["validation_folds"], cfg["seed"] + 1, "any_abnormal"))
    usable["split"] = "train"
    usable.loc[eligible[test], "split"] = "test"
    usable.loc[eligible[dev[val]], "split"] = "val"
    out.mkdir(parents=True, exist_ok=True)
    usable[["record_id", "mother_id", "site", "split", "prior_seen"]].to_csv(out / "records.csv", index=False)
    segments = table(data_dir / "segments.csv")
    segments = segments.merge(usable[["record_id", "split"]], on="record_id", how="left", validate="many_to_one")
    for split in ("train", "val", "test"):
        require_two_classes(segments.loc[segments.split == split, "target"], split)
    segments[["record_id", "mother_id", "seg_idx", "split"]].to_csv(out / "segments.csv", index=False)
    write_json(out / "summary.json", {split: {"records": int((usable.split == split).sum()),
        "mothers": int(usable.loc[usable.split == split, "mother_id"].nunique()),
        "segments": int((segments.split == split).sum()),
        "positives": int(segments.loc[segments.split == split, "target"].sum())} for split in ("train", "val", "test")})
    write_json(out / "cohort_history.json", dict(status="checked" if prior_file else "not_checked",
        policy="previously_seen_mothers_training_only", supplied_mothers=len(known),
        matched_mothers=int(usable.loc[usable.prior_seen, "mother_id"].nunique()),
        matched_validation_mothers=int(usable.loc[usable.prior_seen & usable.split.eq("val"), "mother_id"].nunique()),
        matched_test_mothers=int(usable.loc[usable.prior_seen & usable.split.eq("test"), "mother_id"].nunique()),
        note="No prior cohort file means previous exposure was not checked; it does not establish a new untouched cohort."))


def load_segments(run):
    frame = table(run / "data/segments.csv")
    split = table(run / "splits/segments.csv")
    frame = frame.merge(split[["record_id", "seg_idx", "split"]], on=["record_id", "seg_idx"], how="left", validate="one_to_one", sort=False)
    if frame.split.isna().any():
        raise ValueError("Missing split assignments")
    return frame


def threshold90(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, float)
    require_two_classes(y, "validation operating point")
    negatives = np.sort(scores[y == 0])
    # With >= as the alarm rule, nextafter handles tied scores conservatively.
    return float(np.nextafter(negatives[int(np.ceil(0.9 * len(negatives))) - 1], np.inf))


class Scorer:
    def __init__(self, frame, scores, threshold):
        self.y = frame.target.to_numpy(int)
        self.p = np.asarray(scores, float)
        if len(self.y) != len(self.p) or not np.isfinite(self.p).all():
            raise ValueError("Prediction alignment/finite check failed")
        self.alarm = self.p >= np.asarray(threshold)
        self.order = np.argsort(-self.p, kind="stable")
        self.ends = np.r_[np.flatnonzero(np.diff(self.p[self.order]) != 0), len(self.p) - 1]
        self.mothers, self.group = np.unique(frame.mother_id.astype(str), return_inverse=True)
        self.record, self.record_idx = np.unique(frame.record_id.astype(str), return_inverse=True)
        normal = frame.get("record_all_normal", pd.Series(0, index=frame.index)).to_numpy(bool)
        self.normal_segments = normal
        first = np.unique(self.record_idx, return_index=True)[1]
        self.record_group = self.group[first]
        self.normal_records = normal[first]
        self.record_alarm = np.bincount(self.record_idx, weights=self.alarm, minlength=len(self.record)) > 0

    def compute(self, group_weights=None):
        gw = np.ones(len(self.mothers)) if group_weights is None else group_weights
        w = gw[self.group]
        pos, neg = np.sum(w * self.y), np.sum(w * (1 - self.y))
        if pos == 0 or neg == 0:
            return None
        tp = np.r_[0, np.cumsum((w * self.y)[self.order])[self.ends]]
        fp = np.r_[0, np.cumsum((w * (1 - self.y))[self.order])[self.ends]]
        tpr, fpr = tp / pos, fp / neg
        precision = tp[1:] / np.maximum(tp[1:] + fp[1:], 1e-30)
        auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        ap = np.sum(np.diff(tpr) * precision)
        true_alarm = np.sum(w * self.alarm * self.y)
        false_alarm = np.sum(w * self.alarm * (1 - self.y))
        pred_pos = true_alarm + false_alarm
        rw = gw[self.record_group]
        normal_rec_denom = np.sum(rw * self.normal_records)
        normal_hours = np.sum(w * self.normal_segments) / 12
        return dict(auroc=auc, auprc=ap, prevalence=pos / (pos + neg),
            sensitivity_at_test_spec90=float(tpr[fpr <= 0.1 + 1e-12].max()),
            sensitivity=true_alarm / pos, specificity=1 - false_alarm / neg,
            ppv=true_alarm / pred_pos if pred_pos else np.nan,
            f1=2 * true_alarm / (pos + pred_pos),
            brier=np.sum(w * (self.p - self.y) ** 2) / w.sum(),
            normal_record_alarm_rate=np.sum(rw * self.normal_records * self.record_alarm) / normal_rec_denom if normal_rec_denom else np.nan,
            false_alarms_per_normal_hour=np.sum(w * self.normal_segments * self.alarm) / normal_hours if normal_hours else np.nan)


def cohort_counts(frame):
    return dict(n=len(frame), mothers=int(frame.mother_id.nunique()), positives=int(frame.target.sum()),
        positive_mothers=int(frame.loc[frame.target == 1, "mother_id"].nunique()),
        negative_mothers=int(frame.loc[frame.target == 0, "mother_id"].nunique()))


def evaluate(frame, scores, threshold, n_boot=0, seed=42):
    scorer = Scorer(frame, scores, threshold)
    point = scorer.compute()
    if point is None:
        return {"status": "insufficient_classes", **cohort_counts(frame)}
    values = {k: [] for k in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(len(scorer.mothers), size=len(scorer.mothers)), minlength=len(scorer.mothers))
        result = scorer.compute(weights)
        if result:
            for k, v in result.items():
                if np.isfinite(v):
                    values[k].append(v)
    return dict(status="ok", **cohort_counts(frame), point=point,
                ci95={k: list(np.quantile(v, [0.025, 0.975])) if v else None for k, v in values.items()},
                bootstrap_valid={k: len(v) for k, v in values.items()})


def paired(frame, left, right, n_boot, seed=42):
    a, b = Scorer(frame, left, 0.5), Scorer(frame, right, 0.5)
    pa, pb = a.compute(), b.compute()
    if pa is None or pb is None:
        return {"status": "insufficient_classes", **cohort_counts(frame)}
    keys = ["auroc", "auprc", "brier"]
    values = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        weights = np.bincount(rng.integers(len(a.mothers), size=len(a.mothers)), minlength=len(a.mothers))
        av, bv = a.compute(weights), b.compute(weights)
        if av is not None and bv is not None:
            for k in keys:
                values[k].append(av[k] - bv[k])
    return {"status": "ok", **cohort_counts(frame), "direction": "left_minus_right", "metrics": {
        k: {"difference": pa[k] - pb[k], "ci95": list(np.quantile(values[k], [0.025, 0.975])) if values[k] else None,
            "valid_bootstrap": len(values[k])} for k in keys},
        "interpretation": "A CI containing zero is inconclusive, not proof of equivalence."}
