"""Scientific controls use synthetic data only; no source-data fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.features import CAT18, CAT28
from fg.models import align_unsmoothed, feature_response, logistic, nonlinear_logistic, run_a
from fg.evaluation import evaluate
from tools.validate_run import metric_values


class NonlinearControlTests(unittest.TestCase):
    def test_read_only_metric_audit_detects_inconsistent_report(self):
        frame = pd.DataFrame(dict(record_id=list("abcd"), mother_id=list("abcd"),
                                  target=[0, 0, 1, 1], score=[.1, .4, .2, .8]))
        result = evaluate(frame, frame.score, .5)
        self.assertEqual(metric_values(frame, result, "synthetic"), 1)
        result["point"]["auprc"] -= .1
        with self.assertRaises(AssertionError):
            metric_values(frame, result, "synthetic")

    def test_u_shaped_risk_is_detected_without_test_fitted_knots(self):
        rng = np.random.default_rng(71)
        training = rng.uniform(-3, 3, size=(1000, 1))
        testing = rng.uniform(-3, 3, size=(600, 1))
        train_y = (np.abs(training[:, 0]) > 1.5).astype(int)
        test_y = (np.abs(testing[:, 0]) > 1.5).astype(int)
        linear = logistic().fit(training, train_y)
        spline = nonlinear_logistic().fit(training, train_y)
        linear_auc = roc_auc_score(test_y, linear.predict_proba(testing)[:, 1])
        nonlinear_auc = roc_auc_score(test_y, spline.predict_proba(testing)[:, 1])
        self.assertLess(linear_auc, .60)
        self.assertGreater(nonlinear_auc, .98)
        basis = spline.named_steps["singlefeaturespline"]
        knots = basis.spline_.bsplines_[0].t.copy()
        spline.predict_proba(np.array([[-10000.], [10000.]]))
        np.testing.assert_array_equal(knots, basis.spline_.bsplines_[0].t)

    def test_constant_and_entirely_missing_predictors_are_intercept_only(self):
        y = np.tile([0, 0, 0, 1], 30)
        for value in (0., np.nan):
            with self.subTest(value=value):
                model = nonlinear_logistic().fit(np.full((len(y), 1), value), y)
                scores = model.predict_proba(np.array([[value], [2.], [100.]]))[:, 1]
                np.testing.assert_allclose(scores, y.mean(), atol=1e-4)
                self.assertTrue(model.named_steps["singlefeaturespline"].constant_)


class CohortAndResponseTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({name: np.arange(40, dtype=float) for name in CAT28})
        self.frame["record_id"] = [str(i) for i in range(40)]
        self.frame["mother_id"] = np.repeat(np.arange(20).astype(str), 2)
        self.frame["seg_idx"] = 0
        self.frame["target"] = np.tile([0, 1], 20)
        self.frame["split"] = "train"

    def test_cat18_is_the_exact_original_subset(self):
        self.assertEqual(CAT18, CAT28[:18])
        self.assertEqual(len(CAT18), 18)
        self.assertNotIn("figo_baseline", CAT18)
        self.assertIn("ft_corr_min_lag_s", CAT18)

    def test_sensitivity_aligns_keys_not_file_row_order(self):
        candidate = self.frame.sample(frac=1, random_state=13).copy()
        candidate[CAT28] += 1
        aligned = align_unsmoothed(self.frame, candidate)
        np.testing.assert_array_equal(aligned.record_id, self.frame.record_id)
        np.testing.assert_array_equal(aligned.target, self.frame.target)
        np.testing.assert_array_equal(aligned[CAT28], self.frame[CAT28] + 1)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            align_unsmoothed(self.frame, candidate.iloc[1:])
        candidate.loc[candidate.index[0], "target"] = 1 - candidate.iloc[0].target
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            align_unsmoothed(self.frame, candidate)

    def test_train_response_counts_and_constant_bin(self):
        self.frame["n_severe_decel"] = 0
        response = feature_response(self.frame)
        self.assertFalse({"record_id", "mother_id"} & set(response.columns))
        for feature in CAT28:
            rows = response[response.feature == feature]
            self.assertEqual(rows.n_segments.sum(), len(self.frame))
            self.assertEqual(rows.n_positive.sum(), self.frame.target.sum())
            self.assertTrue((rows.n_mothers >= rows.positive_mothers).all())
            self.assertTrue((rows.n_mothers >= rows.negative_mothers).all())
        self.assertEqual(len(response[response.feature == "n_severe_decel"]), 1)
        self.frame.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "training rows"):
            feature_response(self.frame)

    def test_complete_synthetic_experiment_shares_test_cohort(self):
        rng = np.random.default_rng(4)
        n = 240
        frame = pd.DataFrame(rng.normal(size=(n, len(CAT28))), columns=CAT28)
        frame["record_id"] = [f"toy{i}" for i in range(n)]
        frame["mother_id"] = [f"mother{i}" for i in range(n)]
        frame["seg_idx"] = 0
        frame["target"] = np.tile([0, 1], n // 2)
        frame["record_all_normal"] = 1 - frame.target
        frame["site"] = "toy"
        frame["fhr_mean"] += frame.target
        split = np.repeat(["train", "val", "test"], [120, 60, 60])
        cfg = dict(seed=42, threads=1, budget=dict(seeds=[42, 43], tree_iterations=5,
                   tree_patience=2, cv_folds=2, bootstrap=0, shap_samples=10))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "splits").mkdir()
            frame.to_csv(root / "data/segments.csv", index=False)
            frame.sample(frac=1, random_state=5).to_csv(root / "data/segments_unsmoothed.csv", index=False)
            frame[["record_id", "seg_idx"]].assign(split=split).to_csv(root / "splits/segments.csv", index=False)
            run_a(root, root / "experiment_a", cfg)
            result = json.loads((root / "experiment_a/holdout_metrics.json").read_text())
            expected = {"cat18", "cat28", "logistic28", "best_single", "best_single_nonlinear", "cat28_unsmoothed"}
            self.assertTrue(expected <= set(result))
            self.assertEqual({entry["n"] for entry in result.values()}, {60})
            self.assertEqual(result["cat28"]["point"], result["cat28_unsmoothed"]["point"])
            predictions = pd.read_csv(root / "experiment_a/test_predictions.csv")
            expected_keys = set(frame.loc[split == "test", "record_id"])
            for _, rows in predictions.groupby("model"):
                self.assertEqual(set(rows.record_id), expected_keys)
            seeds = pd.read_csv(root / "experiment_a/seed_holdout_metrics.csv")
            self.assertEqual(seeds[seeds.model == "cat28"].selected.sum(), 1)
            self.assertEqual(set(seeds.seed), {42, 43})
            selection = pd.read_csv(root / "experiment_a/single_feature_selection.csv")
            self.assertEqual(set(selection.family), {"linear", "nonlinear_spline"})
            for family, rows in selection.groupby("family"):
                name = "best_single" if family == "linear" else "best_single_nonlinear"
                self.assertEqual(result[name]["feature"], rows.loc[rows.validation_auroc.idxmax(), "feature"])
            oof = pd.read_csv(root / "experiment_a/cv_predictions_seed42.csv")
            self.assertEqual(sum(c.startswith("score__single_") for c in oof), 28)
            self.assertEqual(sum(c.startswith("score__nonlinear_single_") for c in oof), 28)
            deltas = pd.read_csv(root / "experiment_a/seed_paired_cat28_minus_comparator.csv")
            self.assertEqual(set(deltas.seed), {42, 43})
            self.assertFalse(any("ci" in name for name in deltas.columns))
            for column in ("auroc_difference", "auprc_difference", "brier_difference"):
                np.testing.assert_allclose(deltas.loc[deltas.comparator == "cat28_unsmoothed", column], 0)


if __name__ == "__main__":
    unittest.main()
