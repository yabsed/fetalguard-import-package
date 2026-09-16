"""Synthetic tests for validation-only selection and matched scientific controls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.cnn import capacity_summary, select_capacity, matched_normalization_plan, normalize_holdout, run_b
from fg.data import EMR_FEATURES
from fg.features import CAT28
from fg.supplementary import metadata, outcomes, outcome_cohort_summary


class CapacitySelectionTests(unittest.TestCase):
    def candidates(self):
        rows = []
        values = {
            "channel_maxabs": {8: [.9, .1], 16: [.6, .6], 64: [.55, .55], 128: [.8, .2]},
            "per_segment_z": {8: [.95, .95], 16: [.4, .4], 64: [.3, .3], 128: [.7, .7]},
        }
        for mode, widths in values.items():
            for width, scores in widths.items():
                for seed, score in zip([42, 43], scores):
                    rows.append(dict(normalization=mode, width=width, seed=seed,
                                     validation_auprc=score, hypothetical_test_ap=1 - score))
        return rows

    def test_width_uses_seed_mean_not_lucky_seed_or_test_score(self):
        candidates = self.candidates()
        summary = capacity_summary(candidates, [42, 43])
        chosen = select_capacity(candidates, summary, "channel_maxabs", "small")
        self.assertEqual(chosen["width"], 16)
        self.assertEqual(chosen["seed"], 42)
        for row in candidates:
            row["hypothetical_test_ap"] = 1000 if row["width"] == 8 else -1000
        self.assertEqual(select_capacity(candidates, summary, "channel_maxabs", "small")["width"], 16)

    def test_missing_or_duplicate_seed_must_not_bias_mean(self):
        candidates = self.candidates()
        with self.assertRaisesRegex(ValueError, "seed grid"):
            capacity_summary(candidates[:-1], [42, 43])
        with self.assertRaisesRegex(ValueError, "seed grid"):
            capacity_summary(candidates + [candidates[0]], [42, 43])

    def test_normalization_holds_width_and_every_seed_fixed(self):
        candidates = self.candidates()
        plan = matched_normalization_plan(candidates, capacity_summary(candidates, [42, 43]), [42, 43])
        self.assertEqual([(row["arm"], row["width"], row["seed"]) for row in plan],
                         [("small", 16, 42), ("small", 16, 43), ("medium", 64, 42), ("medium", 64, 43)])
        for row in plan:
            self.assertEqual(row["left"]["width"], row["right"]["width"])
            self.assertEqual(row["left"]["seed"], row["right"]["seed"])

    def test_holdout_normalization_reuses_training_scale(self):
        raw = np.array([[[100., 200.], [10., 20.]]], dtype=np.float32)
        normalized = normalize_holdout(raw, "channel_maxabs", {"scale": [100., 10.]})
        np.testing.assert_array_equal(normalized, [[[1, 2], [1, 2]]])

    def test_small_synthetic_run_writes_matched_predictions_and_costs(self):
        import torch
        torch.set_num_threads(1)
        rows = 64
        frame = pd.DataFrame({
            "record_id": [f"r{i}" for i in range(rows)], "mother_id": [f"m{i}" for i in range(rows)],
            "seg_idx": [0] * rows, "target": [0, 1] * (rows // 2),
            "record_all_normal": [1, 0] * (rows // 2), "site": ["synthetic"] * rows,
            "split": ["train"] * 32 + ["val"] * 16 + ["test"] * 16,
        })
        cfg = {"cnn": True, "resolved_device": "cpu", "seed": 42,
               "budget": {"cnn_norms": ["channel_maxabs", "per_segment_z"], "cnn_widths": [8, 64],
                          "seeds": [42, 43], "cnn_epochs": 2, "cnn_patience": 2, "batch_size": 16, "bootstrap": 4}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "experiment_a").mkdir()
            np.save(root / "data/signals.npy", np.random.default_rng(9).normal(size=(rows, 2, 150)).astype(np.float32))
            frame[frame.split.eq("test")].assign(model="cat28", score=.5).to_csv(root / "experiment_a/test_predictions.csv", index=False)
            out = root / "experiment_b"
            with patch("fg.cnn.load_segments", return_value=frame), patch("fg.cnn.note"):
                run_b(root, out, cfg)
            metrics = json.loads((out / "metrics.json").read_text())
            matched = json.loads((out / "normalization_matched.json").read_text())
            self.assertEqual(len(metrics), 4)
            self.assertEqual(len(matched["comparisons"]), 4)
            self.assertEqual(len(pd.read_csv(out / "normalization_matched_predictions.csv")), 16 * 2 * 2 * 2)
            costs = pd.read_csv(out / "inference_cost.csv")
            self.assertEqual(len(costs), 4)
            self.assertTrue(costs.seconds.gt(0).all())
            self.assertTrue(costs.model_file_bytes.gt(0).all())
            for row in matched["comparisons"]:
                self.assertIn(row["width"], [8, 64])
                self.assertIn(row["seed"], [42, 43])


class MetadataComparisonTests(unittest.TestCase):
    def test_both_arms_use_identical_seed_budget_and_validation_selection(self):
        rows = 16
        frame = pd.DataFrame({
            "record_id": [f"record_{i}" for i in range(rows)], "mother_id": [f"mother_{i}" for i in range(rows)],
            "seg_idx": np.zeros(rows, dtype=int), "target": [0, 1] * (rows // 2),
            "record_all_normal": [1, 0] * (rows // 2), "split": ["train"] * 8 + ["val"] * 4 + ["test"] * 4,
            "site": ["synthetic"] * rows, "maternal_age": [30.] * rows,
            "gestational_age": [38.] * rows, "missing_fraction": [0.] * rows,
        })
        cfg = {"seed": 41, "budget": {"seeds": [41, 42], "bootstrap": 4}}
        calls = []

        def fake_fit(train, val, columns, config, seed, path):
            arm = "cat28" if columns == CAT28 else "with_emr"
            calls.append((arm, seed, len(train), len(val), path.name))
            return arm, seed

        def fake_predict(model, part, columns):
            best = model in {("cat28", 41), ("with_emr", 42)}
            if part.split.eq("val").all():
                return np.array([.1, .8, .2, .9]) if best else np.array([.8, .1, .9, .2])
            # Deliberately invert test quality: it must not influence seed selection.
            return np.array([.8, .1, .9, .2]) if best else np.array([.1, .8, .2, .9])

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            out = run / "supplementary"
            out.mkdir()
            with patch("fg.supplementary.with_metadata", return_value=frame), \
                 patch("fg.supplementary.fit_tree", side_effect=fake_fit), \
                 patch("fg.supplementary.predict", side_effect=fake_predict):
                metadata(run, out, cfg)
            result = json.loads((out / "emr_added.json").read_text())
            self.assertEqual([(arm, seed) for arm, seed, *_ in calls],
                             [("cat28", 41), ("cat28", 42), ("with_emr", 41), ("with_emr", 42)])
            self.assertTrue(all((train, val) == (8, 4) for _, _, train, val, _ in calls))
            self.assertEqual([calls[0][-1], calls[1][-1]], ["holdout_cat28_41.cbm", "holdout_cat28_42.cbm"])
            self.assertEqual(result["selection"]["cat28"]["seed"], 41)
            self.assertEqual(result["selection"]["with_emr"]["seed"], 42)
            self.assertEqual([row["seed"] for row in result["matched_seed_comparisons"]], [41, 42])
            self.assertEqual(result["features"], CAT28 + EMR_FEATURES)


class OutcomeDenominatorTests(unittest.TestCase):
    def test_unique_mothers_are_not_counted_as_records(self):
        frame = pd.DataFrame({"mother_id": ["mother_a", "mother_a", "mother_b"],
                              "any_abnormal": [1, 0, 0], "maternal_age": [30., 30., np.nan],
                              "abnormal_fraction": [.2, 0, 0], "longest_abnormal_run": [np.nan, 0, 0]})
        result = outcome_cohort_summary(frame)
        self.assertEqual((result["records"], result["mothers"]), (3, 2))
        self.assertEqual(result["maternal_age"]["observed_mothers"], 1)
        self.assertEqual(result["longest_abnormal_run"]["missing_records"], 1)
        self.assertNotIn("mother_a", json.dumps(result))

    def test_missingness_and_association_survive_insufficient_outcome_groups(self):
        frame = pd.DataFrame({
            "record_id": ["r1", "r2", "r3", "r4"], "mother_id": ["m1", "m2", "m3", "m4"],
            "any_abnormal": [1, 0, 1, 0], "abnormal_fraction": [.8, 0, .6, 0],
            "longest_abnormal_run": [2., 0, np.nan, 0], "emr_UA.pH": [7.1, 7.3, np.nan, 8.5],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("fg.supplementary.table", return_value=frame), patch("fg.supplementary.fit_tree") as fit:
                outcomes(root, root, {"budget": {"cv_folds": 5}})
            fit.assert_not_called()
            result = json.loads((root / "outcomes.json").read_text())
            ph = result["ph_lt_7_20"]
            self.assertEqual(ph["status"], "insufficient_outcome_groups")
            self.assertEqual(ph["cohorts"]["observed_valid"]["mothers"], 2)
            self.assertEqual(ph["cohorts"]["missing_or_invalid"]["mothers"], 2)
            self.assertEqual(ph["positive_mothers"], 1)
            self.assertEqual(sum(cell["records"] for cell in ph["reading_outcome_cells"]), 2)
            self.assertEqual(ph["expert_abnormal_fraction_auroc"], 1.)
            self.assertEqual(result["apgar1_lt_7"]["status"], "missing_field")


if __name__ == "__main__":
    unittest.main()
