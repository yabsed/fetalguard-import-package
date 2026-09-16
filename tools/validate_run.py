"""Read-only audit of legacy and v2 runs; never writes patient artifacts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.common import read_json, sha256, table
from fg.features import CAT28
from fg.evaluation import load_segments


def predictions(path):
    # Preserve floating-point ties: the default CSV parser can round nearby
    # probabilities differently and slightly change rank-based metrics.
    return pd.read_csv(path, dtype={"record_id": str, "mother_id": str, "site": str},
                       float_precision="round_trip", low_memory=False)


def metric_values(frame, result, label, score="score"):
    """Recompute discrimination with sklearn, independently of our Scorer."""
    if result.get("status") != "ok":
        return 0
    assert len(frame) == result["n"], (label, "metric denominator mismatch")
    assert frame[score].between(0, 1).all(), (label, "invalid probabilities")
    assert frame.target.nunique() == 2, (label, "missing class")
    expected = {"auroc": roc_auc_score(frame.target, frame[score]),
                "auprc": average_precision_score(frame.target, frame[score])}
    for key, value in expected.items():
        assert np.isclose(value, result["point"][key], rtol=1e-10, atol=1e-12), (label, key, value, result["point"][key])
    return 1


def stage_hashes(root, stage):
    marker = read_json(root / ".state" / f"{stage}.json")
    assert marker["files"], (stage, "empty stage marker")
    for name, digest in marker["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts, (stage, "invalid artifact path")
        path = root / stage / relative
        assert path.is_file() and sha256(path) == digest, (stage, name)


def assert_cohort(rows, expected, label):
    keys = ["record_id", "seg_idx"]
    actual_keys = pd.MultiIndex.from_frame(rows[keys]).sort_values()
    expected_keys = pd.MultiIndex.from_frame(expected[keys]).sort_values()
    assert actual_keys.equals(expected_keys), (label, "test cohort differs")
    compared = rows.merge(expected[keys + ["target", "mother_id"]], on=keys,
                          suffixes=("", "_source"), validate="one_to_one")
    assert compared.target.eq(compared.target_source).all(), (label, "target mismatch")
    assert compared.mother_id.eq(compared.mother_id_source).all(), (label, "mother mismatch")


def supplementary_metrics(run, frame):
    checked = 0
    outcome_file = run / "supplementary/outcome_oof_predictions.csv"
    if outcome_file.is_file():
        summaries = read_json(run / "supplementary/outcomes.json")
        prediction = predictions(outcome_file)
        for (outcome, arm), rows in prediction.groupby(["outcome", "arm"]):
            assert not rows.record_id.duplicated().any(), (outcome, arm, "duplicate outcome predictions")
            assert rows.groupby("mother_id").fold.nunique().max() == 1, (outcome, "outcome fold leakage")
            checked += metric_values(rows, summaries[outcome]["metrics"][arm], f"outcome/{outcome}/{arm}")
    loso_file = run / "supplementary/loso_predictions.csv"
    if loso_file.is_file():
        summaries = read_json(run / "supplementary/loso_metrics.json")
        for (site, arm), rows in predictions(loso_file).groupby(["site", "arm"]):
            assert_cohort(rows, frame[frame.site == site], f"loso/{site}/{arm}")
            checked += metric_values(rows, summaries[site][arm], f"loso/{site}/{arm}")
    emr_file = run / "supplementary/emr_predictions.csv"
    if emr_file.is_file():
        rows = predictions(emr_file)
        result = read_json(run / "supplementary/emr_added.json")
        assert_cohort(rows, frame[frame.split == "test"], "emr")
        checked += metric_values(rows, result["metrics"], "emr", score="score_emr")
        if "baseline_metrics" in result:
            checked += metric_values(rows, result["baseline_metrics"], "emr_baseline")
    official_file = run / "official/xgboost_emergency_predictions.csv"
    if official_file.is_file():
        summaries = read_json(run / "official/xgboost_emergency_metrics.json")
        for mode, rows in predictions(official_file).groupby("parser"):
            checked += metric_values(rows, summaries[mode], f"official_xgb/{mode}")
    yolo_file = run / "official/yolo_predictions.csv"
    if yolo_file.is_file():
        result = read_json(run / "official/yolo_metrics.json")
        if "metrics" in result:
            rows = predictions(yolo_file).merge(frame[["record_id", "seg_idx", "split"]],
                on=["record_id", "seg_idx"], validate="one_to_one")
            rows = rows[rows.split == "test"]
            checked += metric_values(rows, result["metrics"], "official_yolo")
            baseline = predictions(run / "experiment_a/test_predictions.csv")
            baseline = baseline[baseline.model == "cat28"]
            matched = rows.merge(baseline[["record_id", "seg_idx", "score"]],
                on=["record_id", "seg_idx"], suffixes=("", "_cat28"), validate="one_to_one")
            checked += metric_values(matched, result["cat28_same_image_subset"], "cat28_yolo_subset", "score_cat28")
    return checked


def validate_run(path, expect_records=None, check_original_xgb=False):
    supplied = Path(path).resolve()
    if (supplied / "internal").is_dir():
        root, run, v2 = supplied, supplied / "internal", True
    elif supplied.name == "internal" and (supplied.parent / "status.json").is_file():
        root, run, v2 = supplied.parent, supplied, True
    else:
        root, run, v2 = supplied, supplied, False
    status = read_json(root / "status.json")
    assert status["status"] == "complete", status
    for stage in ("data", "splits", "experiment_a", "experiment_b", "supplementary", "official", "report"):
        stage_hashes(run, stage)
    summary = read_json(run / "data/summary.json")
    if expect_records is not None:
        assert summary["records"] == expect_records, summary
    records = table(run / "splits/records.csv")
    assert records.groupby("mother_id").split.nunique().max() == 1
    if "prior_seen" in records:
        flags = records.prior_seen.astype(str).str.lower()
        assert flags.isin(["true", "false", "1", "0"]).all(), "Invalid prior_seen flags"
        assert records.loc[flags.isin(["true", "1"]), "split"].eq("train").all(), "Previously seen mothers outside training"
    frame = load_segments(run)
    raw = np.load(run / "data/signals.npy", mmap_mode="r")
    assert raw.shape == (len(frame), 2, 150), raw.shape
    assert np.isfinite(raw).all()
    assert np.isfinite(frame[CAT28]).all().all()
    assert len(frame) == summary["segments"]
    keys = ["record_id", "seg_idx"]
    assert not frame.duplicated(keys).any()
    sensitivity_path = run / "data/segments_unsmoothed.csv"
    if v2:
        assert sensitivity_path.is_file(), "v2 requires smoothing sensitivity features"
    if sensitivity_path.is_file():
        candidate = table(sensitivity_path)
        assert np.isfinite(candidate[CAT28]).all().all()
        assert len(candidate) == len(frame), "Unsmoothed cohort length mismatch"
        merged = frame[keys + ["target"]].merge(candidate, on=keys, how="outer", validate="one_to_one",
            suffixes=("_primary", ""), indicator=True)
        assert merged._merge.eq("both").all(), "Unsmoothed cohort keys mismatch"
        if "target_primary" in merged:
            assert merged.target.eq(merged.target_primary).all(), "Unsmoothed targets mismatch"
    expected = frame[frame.split == "test"]
    checked = 0
    for stage, metrics_file in (("experiment_a", "holdout_metrics.json"), ("experiment_b", "metrics.json")):
        file = run / stage / "test_predictions.csv"
        if not file.exists():
            continue
        metrics = read_json(run / stage / metrics_file)
        pred = predictions(file)
        assert set(pred.model) == set(metrics), (stage, "predictions/metrics models mismatch")
        for model, rows in pred.groupby("model"):
            assert_cohort(rows, expected, f"{stage}/{model}")
            checked += metric_values(rows, metrics[model], f"{stage}/{model}")
    development = set(frame.loc[frame.split != "test", "mother_id"])
    summaries = {(int(row["seed"]), row["model"]): row["result"] for row in read_json(run / "experiment_a/cv_summary.json")}
    for file in (run / "experiment_a").glob("cv_predictions_seed*.csv"):
        seed = int(file.stem.removeprefix("cv_predictions_seed"))
        oof = predictions(file)
        assert set(oof.mother_id) == development
        assert oof.groupby("mother_id").fold.nunique().max() == 1
        assert not oof.isna().any().any()
        assert len([c for c in oof if c.startswith("score__single_")]) == 28
        if v2:
            assert len([c for c in oof if c.startswith("score__nonlinear_single_")]) == 28
        for column in oof:
            if column.startswith("score__"):
                model = column.removeprefix("score__")
                checked += metric_values(oof, summaries[(seed, model)], f"cv/{seed}/{model}", column)
    checked += supplementary_metrics(run, frame)
    figo = table(run / "supplementary/figo_annotation_agreement.csv")
    assert len(figo) == 7
    if check_original_xgb:
        official = read_json(run / "official/xgboost_emergency_metrics.json")
        assert abs(official["original"]["accuracy"] - 1978 / 2301) < 1e-12
    assert (run / "report/report.html").stat().st_size > 1000
    export = None
    if v2:
        stage_hashes(root, "export_review")
        from fg.export_review import validate_review_bundle
        export = validate_review_bundle(root / "export_review")
        banned_columns = {"record_id", "mother_id", "seg_idx", "target", "score", "score_emr", "path", "model_file", "source_path"}
        for file in (root / "export_review").glob("*.csv"):
            columns = pd.read_csv(file, nrows=0).columns
            assert not banned_columns.intersection(columns), (file.name, "individual columns in review bundle")
    return {"status": "PASS", "layout": "v2" if v2 else "legacy", "records": summary["records"], "mothers": summary["mothers"],
            "segments": len(frame), "all_stage_hashes_match": True, "mother_leakage": False,
            "shared_test_cohort": True, "sklearn_metric_pairs_verified": checked,
            "unsmoothed_cohort_verified": sensitivity_path.is_file(), "export_review": export}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Run root, or its internal directory; legacy runs also supported")
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--check-original-xgb", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_run(args.run, args.expect_records, args.check_original_xgb), indent=2))


if __name__ == "__main__":
    main()
