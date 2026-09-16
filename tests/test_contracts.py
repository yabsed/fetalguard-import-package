"""Synthetic-only unit tests. No patient fixtures are bundled."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.data import discover, bbox_check, EMR_FEATURES, OFFICIAL_COLUMNS
from fg.common import write_json
from fg.features import CAT28, GROUPS, extract_segment, interpolate_signal
from fg.feature_rules import detect_decelerations
from fg.evaluation import Scorer, evaluate, paired, threshold90, group_folds
from fg.signal_io import load_case_signal, parse_window_abnormality, DataValidationError
from fg.official import segment_scores, nms_single_class
from fg.cnn import normalize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(record_id=["a", "a", "b", "b", "c", "c", "d", "d"],
            mother_id=["A", "A", "B", "B", "B", "B", "D", "D"], target=[0, 0, 1, 0, 1, 1, 0, 1],
            record_all_normal=[1, 1, 0, 0, 0, 0, 0, 0]))
        self.p = np.array([.2, .5, .5, .2, .8, .8, .5, .9])

    def test_fast_metrics_match_sklearn_with_ties_and_cluster_weights(self):
        scorer = Scorer(self.frame, self.p, .5)
        weights = np.array([2, 3, 1])
        row_weights = weights[scorer.group]
        values = scorer.compute(weights)
        self.assertAlmostEqual(values["auroc"], roc_auc_score(self.frame.target, self.p, sample_weight=row_weights))
        self.assertAlmostEqual(values["auprc"], average_precision_score(self.frame.target, self.p, sample_weight=row_weights))
        self.assertAlmostEqual(values["brier"], brier_score_loss(self.frame.target, self.p, sample_weight=row_weights))

    def test_identical_paired_predictions_have_zero_interval(self):
        result = paired(self.frame, self.p, self.p, 50)
        for metric in result["metrics"].values():
            self.assertEqual(metric["difference"], 0)
            self.assertEqual(metric["ci95"], [0, 0])

    def test_operating_point_is_conservative_for_ties(self):
        y = np.r_[np.zeros(20), np.ones(5)]
        p = np.r_[np.repeat(.8, 20), np.repeat(.9, 5)]
        t = threshold90(y, p)
        self.assertGreaterEqual(np.mean(p[y == 0] < t), .9)
        self.assertEqual(np.mean(p[y == 1] >= t), 1)

    def test_normal_record_alarm_and_hour_denominator(self):
        point = Scorer(self.frame, self.p, .5).compute()
        self.assertEqual(point["normal_record_alarm_rate"], 1)
        self.assertEqual(point["false_alarms_per_normal_hour"], 6)

    def test_mother_grouping_keeps_twins_and_repeats_together(self):
        frame = pd.DataFrame(dict(mother_id=np.repeat(np.arange(40).astype(str), 3), target=np.tile([0, 1, 0], 40)))
        seen = []
        for tr, te in group_folds(frame, 5, 42):
            self.assertFalse(set(frame.iloc[tr].mother_id) & set(frame.iloc[te].mother_id))
            seen.extend(te)
        self.assertEqual(sorted(seen), list(range(len(frame))))

    def test_single_class_is_explicit_not_fake_auc(self):
        frame = self.frame.copy()
        frame["target"] = 0
        self.assertEqual(evaluate(frame, self.p, .5, 50)["status"], "insufficient_classes")


class SignalTests(unittest.TestCase):
    def test_28_features_and_group_partition(self):
        x = np.arange(150)
        features = extract_segment(140 + 5 * np.sin(x / 6), 10 + 5 * np.cos(x / 10), .5)
        self.assertEqual(list(features), CAT28)
        self.assertEqual(len(features), 28)
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertEqual(sorted(c for cols in GROUPS.values() for c in cols), sorted(CAT28))
        self.assertEqual(features["n_decel"], sum(features[f"n_{k}_decel"] for k in ("early", "late", "prolonged", "severe")))

    def test_15_second_boundary_uses_ceil_at_half_hz(self):
        fhr = np.full(150, 140.0)
        fhr[20:27] = 100
        self.assertEqual(detect_decelerations(fhr, 140, .5), [])
        fhr[27] = 100
        self.assertEqual(detect_decelerations(fhr, 140, .5), [(20, 28)])

    def test_twins_select_only_target_fetus(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = [{"partial_image": f"toy_twin{t}_p{p}.png", "interval_sec": 2,
                      "fhr": ",".join([str(100 + t)] * 150), "toco": ",".join(["-2"] * 150)}
                     for p in range(2) for t in (1, 2)]
            write_json(Path(tmp) / "42_2.json", {"code": "toy_42", "twins": "True", "data": parts})
            case = load_case_signal("42_2", tmp)
            self.assertEqual(case.selected_segment_count, 2)
            np.testing.assert_array_equal(case.fhr, np.full(300, 102))
            np.testing.assert_array_equal(case.toco, np.full(300, -2))
            np.testing.assert_array_equal(parse_window_abnormality("0_1", case), [0, 1])
            with self.assertRaises(DataValidationError):
                parse_window_abnormality("0", case)

    def test_raw_cnn_zero_toco_is_preserved(self):
        x = interpolate_signal(np.array([140, 0, 142]), np.array([0, -1, 0]))
        np.testing.assert_array_equal(x, [[140, 141, 142], [0, -1, 0]])

    def test_scale_is_fitted_only_on_training(self):
        raw = np.array([[[100., 150.], [10., 20.]], [[1000., 1500.], [100., 200.]]], dtype=np.float32)
        x, params = normalize(raw, np.array([True, False]), "channel_maxabs")
        self.assertEqual(params["scale"], [150., 20.])
        self.assertEqual(x[1].max(), 10)

    def test_postnatal_targets_are_not_primary_metadata_features(self):
        self.assertEqual(len(OFFICIAL_COLUMNS), 44)
        self.assertFalse(any("APGAR" in c or "UA." in c or "Delivery" in c or "Emergency" in c for c in EMR_FEATURES))


class DiscoveryTests(unittest.TestCase):
    def test_duplicate_sources_deduplicate_and_detect_box_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for prefix in ("Training", "copy"):
                write_json(root / prefix / "annotation_person/42_0.json", {"data": []})
                write_json(root / prefix / "labels/42_0.json", {"ID": "42_0", "Abnormality": "1", "Bbox": [[0, 0, 100, 100, 100, 100]]})
            catalog, files, duplicates = discover(root)
            self.assertEqual(len(catalog["labels"]), 1)
            self.assertEqual(duplicates, 2)
            write_json(root / "copy/labels/42_0.json", {"ID": "42_0", "Abnormality": "1", "Bbox": [[0, 0, 90, 100, 100, 100]]})
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                discover(root)

    def test_missing_image_is_unverified(self):
        row = bbox_check("42", {"Abnormality": "1", "Bbox": [[0, 0, 100, 100, 100, 100]]}, None)
        self.assertFalse(row["checked"])
        self.assertEqual(row["reason"], "image_missing")

    def test_bbox_validation_uses_actual_dimensions(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.png"
            Image.new("RGB", (200, 100)).save(path)
            good = {"Abnormality": "0_1", "Bbox": [[100, 0, 200, 100, 200, 100]]}
            self.assertTrue(bbox_check("42", good, path)["valid"])
            good["Bbox"][0][0] = 90
            self.assertFalse(bbox_check("42", good, path)["valid"])


class OfficialTests(unittest.TestCase):
    def test_nms_uses_objectness_times_class_score(self):
        pred = np.array([[20, 20, 10, 10, .9, .8], [20, 20, 10, 10, .8, .8], [90, 20, 10, 10, .9, .9]])
        boxes = nms_single_class(pred)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(boxes[:, -1], [.81, .72])

    def test_right_edge_maps_to_last_segment(self):
        boxes = np.array([[95, 0, 105, 5, .8], [0, 0, 20, 5, .2]])
        np.testing.assert_allclose(segment_scores(boxes, 100, 2), [.2, .8])


class RunSafetyTests(unittest.TestCase):
    def test_cnn_epoch_resume_matches_uninterrupted_training(self):
        from unittest.mock import patch
        import torch
        from fg import cnn
        torch.set_num_threads(1)
        cfg = {"resolved_device": "cpu", "budget": {"cnn_epochs": 3, "cnn_patience": 3, "batch_size": 16}}
        frame = pd.DataFrame({"split": ["train"] * 32 + ["val"] * 16, "target": [0, 1] * 24})
        x = np.random.default_rng(5).normal(size=(48, 2, 150)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whole, _ = cnn.train_one(x, frame, 8, "test", 42, root / "whole", cfg)
            original_save = cnn.save_state
            def interrupted(path, payload):
                original_save(path, payload)
                if path.name == "last.pt" and payload["next_epoch"] == 1:
                    raise RuntimeError("simulated interruption")
            with patch("fg.cnn.save_state", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    cnn.train_one(x, frame, 8, "test", 42, root / "resumed", cfg)
            resumed, _ = cnn.train_one(x, frame, 8, "test", 42, root / "resumed", cfg)
            for name, tensor in whole.state_dict().items():
                torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)

    def test_bundle_hash_detects_modified_code(self):
        from unittest.mock import patch
        import run
        from fg.common import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "toy.json", {"version": 1})
            write_json(root / "PACKAGE_MANIFEST.json", {"files": {"toy.json": sha256(root / "toy.json")}})
            with patch.object(run, "PACKAGE", root):
                run.verify_package()
                write_json(root / "toy.json", {"version": 2})
                with self.assertRaisesRegex(ValueError, "무결성"):
                    run.verify_package()

    def test_completed_stage_is_reused_and_tampering_is_not_silent(self):
        from run import run_stage
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def action(out):
                calls.append(1)
                write_json(out / "result.json", {"answer": 42})
            run_stage(root, "toy", action)
            run_stage(root, "toy", action)
            self.assertEqual(calls, [1])
            write_json(root / "toy/result.json", {"answer": 43})
            with self.assertRaisesRegex(ValueError, "변경/삭제"):
                run_stage(root, "toy", action)

    def test_stage_failure_never_leaves_success_marker(self):
        from run import run_stage
        with tempfile.TemporaryDirectory() as tmp:
            def fail(out):
                write_json(out / "partial.json", {"partial": True})
                raise RuntimeError("test failure")
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                run_stage(Path(tmp), "toy", fail)
            self.assertFalse((Path(tmp) / ".state/toy.json").exists())

    def test_jupyter_form_renders_without_dataset_or_code_edits(self):
        from unittest.mock import patch
        from launcher_ui import launch
        with patch("IPython.display.display") as display:
            launch()
        panel = display.call_args.args[0]
        self.assertEqual(panel.children[1].description, "데이터 폴더")
        self.assertEqual(panel.children[4].children[1].description, "전체 실행 / 이어서 실행")

    def test_runtime_network_guard(self):
        from run import forbid_network
        with self.assertRaisesRegex(RuntimeError, "Offline"):
            forbid_network("socket.connect", ())
        self.assertIsNone(forbid_network("open", ()))


if __name__ == "__main__":
    unittest.main()
