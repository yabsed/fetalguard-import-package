import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.survey import run_survey, signal_stats, write_csv
from fg.telemetry import Visit, emit, save_json, scope
from fg.visit_audit import budget_diagnosis, build_visit_audit
from tools.build_visit_audit import build_existing


class SurveyTests(unittest.TestCase):
    def test_raw_values_are_not_imputed_or_given_invented_meaning(self):
        stats = signal_stats("0,0,130,130,130,-1,bad,nan,0")
        self.assertEqual(stats["samples"], 9)
        self.assertEqual(stats["invalid"], 2)
        self.assertEqual(stats["zeros"], 3)
        self.assertEqual(stats["longest_zero_samples"], 2)
        self.assertEqual(stats["longest_flat_nonzero_samples"], 3)
        self.assertEqual(stats["negative"], 1)
        self.assertEqual(signal_stats([True, None])["invalid"], 2)

    def test_malformed_files_variants_links_duplicates_and_missing_fields_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            signal = dict(code="S_001", data=[dict(fhr=[0, 0, 120], toco=[0, 1], interval_sec=1)])
            save_json(root / "annotation_person/S_001.json", signal)
            save_json(root / "annotation_person/S_002.json", signal)
            save_json(root / "copy/annotation_person/S_001.json", signal)
            save_json(root / "labels/S_001.json", dict(ID="S_001", extra=None))
            save_json(root / "labels/orphan.json", dict(ID="orphan"))
            save_json(root / "emr/S_001.json", {"Mother.de-identification_ID": "mom", "value": 9999})
            bad = root / "labels/broken.json"
            bad.write_text("{broken", encoding="utf-8")
            save_json(root / "annotation_person/weird.json", dict(data="unsupported"))
            before = {p: p.read_bytes() for p in root.rglob("*.json")}
            out = Path(tmp) / "survey"
            result = run_survey(root, out)
            self.assertEqual(result["issues"], 2)
            self.assertEqual(result["source_windows"], 3)
            self.assertEqual(result["shape_contract_candidates"], 0)
            self.assertEqual(result["links"]["labels_without_annotation"], 2)
            self.assertEqual(result["duplicate_id_files"]["annotation_person"], 1)
            self.assertEqual(result["exact_source_signal_duplicate_groups"], 1)
            fields = list(csv.DictReader((out / "fields.csv").read_text().splitlines()))
            extra = next(row for row in fields if row["kind"] == "labels" and row["field"] == "extra")
            self.assertEqual(extra["parsed_file_denominator"], "2")
            self.assertEqual(extra["absent"], "1")
            self.assertEqual(extra["null_or_blank"], "1")
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertTrue(all(row["sha256"] for row in csv.DictReader((out / "source_inventory.csv").read_text().splitlines())))


class AuditTests(unittest.TestCase):
    def test_posthoc_defaults_to_export_and_preserves_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "old/internal"
            save_json(run / "run_manifest.json", dict(config=dict(data_root=str(Path(tmp) / "source"))))
            with self.assertRaisesRegex(ValueError, "Inside a run"):
                build_existing(run, run.parent / "export_review/new_audit")
            result = build_existing(run)
            self.assertTrue(result.is_file())
            self.assertEqual(result, run.parent / "export_review/visit_audit/index.html")
            self.assertEqual(build_existing(run), result)

    def test_cap_is_not_convergence_and_early_stop_is_separate(self):
        row = budget_diagnosis(1000, 861, 1000, 150)
        self.assertTrue(row["hit_cap"])
        self.assertFalse(row["patience_met"])
        self.assertEqual(row["stale_iterations"], 139)
        self.assertEqual(row["stop_reason"], "budget_cap")
        self.assertEqual(budget_diagnosis(200, 50, 1000, 150)["stop_reason"], "early_stopping")
        self.assertTrue(budget_diagnosis(1000, 850, 1000, 150)["patience_met"])
        self.assertEqual(budget_diagnosis(1000, 861, 1000, 150, False)["stop_reason"], "incomplete_or_running")
        self.assertEqual(budget_diagnosis(None, None, 1000, 80)["stop_reason"], "uncollected")

    def test_legacy_postprocessing_has_no_test_reads_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, out = Path(tmp) / "old/internal", Path(tmp) / "new_audit"
            save_json(run / "run_manifest.json", dict(config=dict(profile="full", budget=dict(cnn_epochs=4, cnn_patience=3, tree_iterations=1000, tree_patience=80))))
            job = run / "experiment_b/models/toy"
            save_json(job / "complete.json", dict(epochs_run=4, best_epoch=3, seed=42))
            write_csv(job / "history.csv", [dict(epoch=i, train_loss=1/i, validation_auprc=i*.1) for i in range(1, 5)], ["epoch", "train_loss", "validation_auprc"])
            tree = run / "experiment_a/models/holdout_cat28_42.cbm"
            tree.parent.mkdir(parents=True)
            tree.write_bytes(b"not a loadable model")
            # Malformed test predictions would fail if diagnostic selection tried to read them.
            (tree.parent.parent / "predictions.csv").write_bytes(b"\xff")
            before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
            index = build_visit_audit(run, out)
            self.assertTrue(index.is_file())
            rows = list(csv.DictReader((out / "training_diagnostics.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(next(row for row in rows if row["family"] == "tree")["stop_reason"], "uncollected")
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["evidence_errors"], [])
            self.assertFalse(any("predictions" in key for key in summary["source_files"]))
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            (out / "questions.md").write_text("기관 담당자의 답변", encoding="utf-8")
            build_visit_audit(run, out)
            self.assertEqual((out / "questions.md").read_text(), "기관 담당자의 답변")

    def test_corrupt_history_is_reported_not_interpreted_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, out = Path(tmp) / "internal", Path(tmp) / "audit"
            save_json(run / "experiment_b/models/bad/complete.json", {})
            history = run / "experiment_b/models/bad/history.csv"
            history.write_bytes(b"\xff")
            build_visit_audit(run, out)
            result = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(result["evidence_errors"]), 1)
            self.assertEqual(result["stop_counts"], {"incomplete_or_running": 1})

    def test_journal_flushes_failure_and_restores_streams(self):
        stdout = sys.stdout
        with tempfile.TemporaryDirectory() as tmp, patch("fg.telemetry.resource_sample", return_value=dict(process_cpu_seconds=0)):
            visit = Visit(tmp, dict(profile="mock"), interval=15)
            with self.assertRaisesRegex(RuntimeError, "private failure"):
                with visit:
                    print("journal console test")
                    with scope("training"):
                        emit("fit_started", job="example", seed=42)
                        raise RuntimeError("private failure")
            self.assertIs(sys.stdout, stdout)
            self.assertFalse(visit.thread.is_alive())
            rows = [json.loads(line) for line in (visit.path / "events.jsonl").read_text().splitlines()]
            self.assertIn("stage_failed", [row["event"] for row in rows])
            self.assertEqual(rows[-1]["status"], "failed")
            self.assertIn("journal console test", (visit.path / "console.log").read_text())
            self.assertTrue((visit.path / "index.html").is_file())

    def test_core_only_keeps_log_without_rendering_audit(self):
        with tempfile.TemporaryDirectory() as tmp, patch("fg.telemetry.resource_sample", return_value=dict(process_cpu_seconds=0)):
            visit = Visit(tmp, dict(profile="full", audit=False), interval=15, mode="core_only")
            with visit:
                print("core-only log")
            latest = json.loads((Path(tmp) / "LATEST_VISIT.json").read_text())
            self.assertEqual(visit.path.name, "run_log")
            self.assertEqual(latest["scope"], "log_only")
            self.assertIsNone(latest["audit"])
            self.assertFalse((visit.path / "index.html").exists())
            self.assertIn("core-only log", (visit.path / "console.log").read_text())


class TreeHistoryTests(unittest.TestCase):
    def test_actual_tree_iterations_are_logged_and_resume_does_not_rewrite(self):
        import numpy as np
        import pandas as pd
        from fg.models import fit_tree
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(dict(x=rng.normal(size=90), target=[0, 1] * 45))
        cfg = dict(threads=1, budget=dict(tree_iterations=8, tree_patience=3))
        with tempfile.TemporaryDirectory() as tmp:
            for family, suffix in (("cat", "cbm"), ("xgb", "json")):
                path = Path(tmp) / ("model." + suffix)
                model = fit_tree(frame[:60], frame[60:], ["x"], cfg, 42, family, path)
                diagnostic = path.with_name(path.name + ".training.json")
                before = diagnostic.read_bytes()
                info = json.loads(before)
                with path.with_name(path.name + ".history.csv").open() as stream:
                    history = list(csv.DictReader(stream))
                self.assertEqual(info["iterations_run"], len(history))
                self.assertIn(info["score_column"], history[0])
                self.assertLessEqual(info["best_iteration"], info["iterations_run"])
                if family == "cat":
                    self.assertGreaterEqual(info["iterations_run"], model.tree_count_)
                fit_tree(frame[:60], frame[60:], ["x"], cfg, 42, family, path)
                self.assertEqual(before, diagnostic.read_bytes())


if __name__ == "__main__":
    unittest.main()
