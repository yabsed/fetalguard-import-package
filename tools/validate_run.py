"""Read-only assertions for a completed run; no dataset included in this tool."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.common import read_json, sha256, table
from fg.features import CAT28
from fg.evaluation import load_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--check-original-xgb", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    status = read_json(root / "status.json")
    assert status["status"] == "complete", status
    for stage in ("data", "splits", "experiment_a", "experiment_b", "supplementary", "official", "report"):
        marker = read_json(root / ".state" / f"{stage}.json")
        for name, digest in marker["files"].items():
            assert sha256(root / stage / name) == digest, (stage, name)
    summary = read_json(root / "data/summary.json")
    if args.expect_records:
        assert summary["records"] == args.expect_records, summary
    records = table(root / "splits/records.csv")
    assert records.groupby("mother_id").split.nunique().max() == 1
    frame = load_segments(root)
    raw = np.load(root / "data/signals.npy", mmap_mode="r")
    assert raw.shape == (len(frame), 2, 150), raw.shape
    assert np.isfinite(raw).all()
    assert np.isfinite(frame[CAT28]).all().all()
    assert len(frame) == summary["segments"]
    assert not frame.duplicated(["record_id", "seg_idx"]).any()
    keys = ["record_id", "seg_idx"]
    expected = pd.MultiIndex.from_frame(frame.loc[frame.split == "test", keys]).sort_values()
    for file in (root / "experiment_a/test_predictions.csv", root / "experiment_b/test_predictions.csv"):
        if not file.exists():
            continue
        pred = table(file)
        for model, rows in pred.groupby("model"):
            actual = pd.MultiIndex.from_frame(rows[keys]).sort_values()
            assert actual.equals(expected), (file, model, "test cohort differs")
            assert rows.score.between(0, 1).all()
            assert rows.merge(frame[keys + ["target"]], on=keys, suffixes=("", "_source"), validate="one_to_one").eval("target == target_source").all()
    development = set(frame.loc[frame.split != "test", "mother_id"])
    for file in (root / "experiment_a").glob("cv_predictions_seed*.csv"):
        oof = table(file)
        assert set(oof.mother_id) == development
        assert oof.groupby("mother_id").fold.nunique().max() == 1
        assert not oof.isna().any().any()
        assert len([c for c in oof if c.startswith("score__single_")]) == 28
    figo = table(root / "supplementary/figo_annotation_agreement.csv")
    assert len(figo) == 7
    if args.check_original_xgb:
        official = read_json(root / "official/xgboost_emergency_metrics.json")
        assert abs(official["original"]["accuracy"] - 1978 / 2301) < 1e-12
    assert (root / "report/report.html").stat().st_size > 1000
    assert len(list((root / "report/figures").glob("*.png"))) >= 4
    print(json.dumps({"status": "PASS", "records": summary["records"], "mothers": summary["mothers"],
                      "segments": len(frame), "all_stage_hashes_match": True, "mother_leakage": False,
                      "shared_test_cohort": True}, indent=2))


if __name__ == "__main__":
    main()
