"""Fault injection: record atomicity, independent stages, truthful completion."""
import errno
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from fg.common import write_json, read_json
from fg.data import prepare, discover
from fg.resilience import Issues


class ResilienceTests(unittest.TestCase):
    def config(self):
        cfg = read_json(run.PACKAGE / "config.json")
        cfg.update(profile="mock", resolved_device="cpu", threads=1)
        cfg["budget"] = cfg["profiles"]["mock"]
        return cfg

    def record(self, root, rid, n=4, labels=4, bad_last=False):
        parts = []
        for i in range(n):
            signal = 140 + 5 * np.sin(np.arange(150) / 6)
            if bad_last and i == n - 1:
                signal[0] = -1
            parts.append(dict(partial_image=f"{rid}_p{i}.png", interval_sec=2,
                              fhr=signal.tolist(), toco=(10 + np.sin(np.arange(150))).tolist()))
        write_json(root / "annotation_person" / (rid + ".json"), dict(code="site_" + rid, twins=False, data=parts))
        write_json(root / "labels" / (rid + ".json"), dict(ID=rid, Abnormality="_".join(str(i % 2) for i in range(labels)), Emergency=0))
        write_json(root / "emr" / (rid + ".json"), {"ID": rid, "Mother.de-identification_ID": "mother_" + rid})

    def test_mismatched_record_is_excluded_not_truncated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.record(root / "source", "bad", labels=5)
            self.record(root / "source", "good")
            catalog, _, _ = discover(root / "source")
            prepare(catalog, root / "data", self.config())
            records = pd.read_csv(root / "data/records.csv")
            self.assertEqual(records.record_id.tolist(), ["good"])
            self.assertEqual(np.load(root / "data/signals.npy").shape, (4, 2, 150))
            excluded = pd.read_csv(root / "data/record_exclusions.csv")
            self.assertIn("5 Abnormality labels for 4", excluded.iloc[0].reason)
            self.assertEqual(read_json(root / "data/summary.json")["excluded_records"], 1)

    def test_late_record_failure_rolls_back_all_signal_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.record(root / "source", "a_bad", bad_last=True)
            self.record(root / "source", "z_good")
            catalog, _, _ = discover(root / "source")
            prepare(catalog, root / "data", self.config())
            for name in ("segments", "segments_unsmoothed", "records"):
                self.assertEqual(set(pd.read_csv(root / f"data/{name}.csv").record_id), {"z_good"})
            self.assertEqual(np.load(root / "data/signals.npy").shape, (4, 2, 150))

    def test_all_invalid_still_produces_empty_typed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.record(root / "source", "bad", labels=5)
            catalog, _, _ = discover(root / "source")
            prepare(catalog, root / "data", self.config())
            self.assertEqual(np.load(root / "data/signals.npy").shape, (0, 2, 150))
            self.assertEqual(read_json(root / "data/summary.json")["records"], 0)
            self.assertIn("record_id", pd.read_csv(root / "data/records.csv"))

    def test_interrupt_permission_and_full_disk_are_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temp:
            issues = Issues(temp)
            for error in (KeyboardInterrupt(), PermissionError("denied"), OSError(errno.ENOSPC, "full")):
                with self.assertRaises(type(error)):
                    with issues.guard("stop"):
                        raise error

    def test_failed_model_does_not_block_cnn_reports_or_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            internal = root / "internal"
            internal.mkdir()
            visited = []
            def stage(owner, name, action, **kwargs):
                visited.append(name)
                if name == "experiment_a":
                    raise ValueError("private-record-123")
                return {"status": "complete"}
            visit = SimpleNamespace(refresh=lambda: None)
            with patch.object(run, "run_stage", side_effect=stage):
                run.execute_analysis(internal, root, self.config(), {}, visit)
            self.assertEqual(visited, ["data", "splits", "experiment_a", "experiment_b", "supplementary", "official", "report", "export_review", "onsite_figures"])
            value = read_json(root / "status.json")
            self.assertEqual(value["status"], "complete_with_issues")
            self.assertNotIn("private-record", json.dumps(value))
            self.assertEqual(visit.analysis_status, "complete_with_issues")

    def test_core_only_skips_cnn_official_and_audit_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            internal = root / "internal"
            internal.mkdir()
            cfg = self.config()
            cfg.update(core_only=True, audit=False, cnn=False, official_models=False)
            visited = []

            def stage(owner, name, action, **kwargs):
                visited.append(name)
                return {"status": "complete"}

            visit = SimpleNamespace(refresh=lambda: None)
            with patch.object(run, "run_stage", side_effect=stage):
                run.execute_analysis(internal, root, cfg, {}, visit)
            self.assertEqual(visited, ["data", "splits", "experiment_a", "supplementary", "report"])
            status = read_json(root / "status.json")
            self.assertTrue(status["core_only"])
            self.assertEqual(status["export_status"], "disabled_by_core_only")
            self.assertEqual(status["stages"]["experiment_b"]["status"], "skipped_by_core_only")
            self.assertEqual(status["stages"]["official"]["status"], "skipped_by_core_only")
            self.assertEqual(status["stages"]["export_review"]["status"], "skipped_by_core_only")
            self.assertFalse((root / "export_review").exists())

    def test_no_training_data_reaches_real_html_and_screened_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            internal = root / "mock-test/internal"
            internal.mkdir(parents=True)
            cfg = self.config()
            cfg["official_models"] = False
            write_json(internal / "run_manifest.json", dict(config=cfg))
            self.record(root / "source", "bad", labels=5)
            catalog, _, _ = discover(root / "source")
            visit = SimpleNamespace(refresh=lambda: None)
            run.execute_analysis(internal, internal.parent, cfg, catalog, visit)
            value = read_json(internal.parent / "status.json")
            self.assertEqual(value["status"], "complete_with_issues")
            self.assertTrue((internal / "report/report.html").is_file())
            review = internal.parent / "export_review"
            self.assertTrue((review / "report.html").is_file(), value)
            self.assertTrue(list((review / "images").glob("*.png")), value)
            from fg.export_review import validate_review_bundle
            validate_review_bundle(review)
            from tools.validate_run import validate_run
            self.assertEqual(validate_run(internal.parent)["status"], "PARTIAL_VALIDATED")

    def test_one_cnn_capacity_failure_does_not_discard_other_capacity(self):
        from fg import cnn
        from fg.ctgnet import CTGNetMini
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "splits").mkdir()
            frame = pd.DataFrame(dict(record_id=[str(i) for i in range(30)],
                                      mother_id=[str(i) for i in range(30)], seg_idx=0,
                                      target=[i % 2 for i in range(30)], site="toy", record_all_normal=0))
            frame.to_csv(root / "data/segments.csv", index=False)
            frame[["record_id", "mother_id", "seg_idx"]].assign(split=["train"] * 10 + ["val"] * 10 + ["test"] * 10).to_csv(root / "splits/segments.csv", index=False)
            np.save(root / "data/signals.npy", np.random.default_rng(42).normal(size=(30, 2, 150)).astype(np.float32))
            cfg = self.config()
            def training(x, frame, width, mode, seed, out, cfg):
                if width == 64:
                    raise RuntimeError("synthetic device failure")
                out.mkdir(parents=True)
                model = CTGNetMini(width=width, input_len=150)
                torch.save(model.state_dict(), out / "best.pt")
                return model, dict(width=width, normalization=mode, seed=seed, parameters=1,
                                   validation_auprc=0.5, hit_cap=True)
            with patch.object(cnn, "train_one", side_effect=training):
                cnn.run_b(root, root / "experiment_b", cfg)
            metrics = read_json(root / "experiment_b/metrics.json")
            self.assertEqual(set(metrics), {"cnn_small_channel_maxabs", "cnn_small_per_segment_z"})
            self.assertEqual(read_json(root / "experiment_b/status.json")["status"], "complete_with_issues")

    def test_official_xgb_failure_still_attempts_yolo(self):
        from fg import official
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(official, "_run_xgb", side_effect=ValueError("synthetic failure")), patch.object(official, "_run_yolo") as yolo:
                official.run_official(run.PACKAGE, out, out, self.config(), {})
            yolo.assert_called_once()
            self.assertEqual(read_json(out / "status.json")["status"], "complete_with_issues")


if __name__ == "__main__":
    unittest.main()
