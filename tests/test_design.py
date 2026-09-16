"""Scientific contracts: temporal continuity, measurement arms, prior exposure."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.data import longest_source_run, feature_quality
from fg.feature_rules import preprocess_for_features
from fg.features import CAT28
from fg.evaluation import create_splits, evaluate


class DesignTests(unittest.TestCase):
    def test_reading_run_resets_at_recording_boundary_and_missing_index(self):
        names = ["toy_1_p000.png", "toy_1_p001.png", "toy_2_p000.png", "toy_2_p002.png"]
        self.assertEqual(longest_source_run([1, 1, 1, 1], names), (2, "source_index_proxy"))
        self.assertEqual(longest_source_run([1, 0, 1, 1], names)[0], 1)
        result, status = longest_source_run([1, 1], ["unknown.png", "other.png"])
        self.assertTrue(np.isnan(result))
        self.assertEqual(status, "unknown_source_order")

    def test_smoothing_sensitivity_keeps_gap_policy_and_only_changes_filter(self):
        x = np.full(150, 140.0)
        x[40:48] = 100
        x[10:13] = 0
        t = np.full(150, 10.0)
        a, ta = preprocess_for_features(x, t, .5)
        b, tb = preprocess_for_features(x, t, .5, smooth_seconds=0)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        self.assertEqual(b[10], 140)
        self.assertEqual(b[40], 100)
        self.assertGreater(a[40], b[40])
        np.testing.assert_allclose(ta, tb)
        with self.assertRaises(ValueError):
            preprocess_for_features(x, t, .5, smooth_seconds=15)

    def test_unobservable_feature_not_reported_as_absent_clinical_event(self):
        frame = pd.DataFrame({k: [0., 0., 0., 0.] for k in CAT28})
        frame = frame.assign(mother_id=["a", "a", "b", "c"], target=[0, 1, 0, 1])
        q = feature_quality(frame).set_index("feature")
        self.assertEqual(q.loc["n_severe_decel", "status"], "structurally_unobservable")
        self.assertEqual(q.loc["stv", "definition_note"], "sample_difference_not_clinical_stv")
        self.assertEqual(q.loc["stv", "zero_mothers"], 3)

    def test_previously_exposed_mothers_are_training_only_including_twins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            records = pd.DataFrame({"record_id": [str(i) for i in range(200)],
                "mother_id": [str(i // 2) for i in range(200)], "site": "toy", "n_kept": 1,
                "any_abnormal": [i // 2 % 2 for i in range(200)]})
            records.to_csv(data / "records.csv", index=False)
            records[["record_id", "mother_id", "site"]].assign(seg_idx=0, target=records.any_abnormal).to_csv(data / "segments.csv", index=False)
            prior = root / "prior.csv"
            pd.DataFrame({"mother_id": [str(i) for i in range(10)]}).to_csv(prior, index=False)
            cfg = {"prior_cohort_file": str(prior), "test_folds": 10, "validation_folds": 9, "seed": 42}
            create_splits(data, root / "split", cfg)
            split = pd.read_csv(root / "split/records.csv", dtype={"mother_id": str})
            self.assertTrue(split.loc[split.prior_seen, "split"].eq("train").all())
            self.assertEqual(split.groupby("mother_id").split.nunique().max(), 1)
            records[["mother_id"]].to_csv(prior, index=False)
            with self.assertRaisesRegex(ValueError, "previously unseen"):
                create_splits(data, root / "all_seen", cfg)

    def test_metric_counts_are_mothers_not_independent_segments(self):
        frame = pd.DataFrame({"record_id": ["a", "a", "b", "c"], "mother_id": ["A", "A", "B", "C"],
                              "target": [0, 1, 1, 0]})
        result = evaluate(frame, [.1, .8, .9, .2], .5)
        self.assertEqual(result["mothers"], 3)
        self.assertEqual(result["positive_mothers"], 2)
        self.assertEqual(result["negative_mothers"], 2)


if __name__ == "__main__":
    unittest.main()
