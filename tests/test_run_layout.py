"""Runner integration checks for private errors and preflight routing."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run


class RunLayoutTests(unittest.TestCase):
    def configured_runner(self, temporary, check):
        root = Path(temporary)
        (root / "source").mkdir()
        cfg = json.loads((run.PACKAGE / "config.json").read_text())
        cfg.update(profile="mock", output_root=str(root), data_root=str(root / "source"),
                   resolved_device="cpu", threads=1)
        cfg["budget"] = cfg["profiles"]["mock"]
        lock = Mock()
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ))
        stack.enter_context(patch.object(run, "config_args", return_value=(SimpleNamespace(check=check), cfg)))
        stack.enter_context(patch.object(run, "verify_package", return_value="synthetic-package-hash"))
        stack.enter_context(patch.object(run, "preflight", return_value={"device": "cpu"}))
        stack.enter_context(patch.object(run.sys, "addaudithook"))
        stack.enter_context(patch.object(run, "acquire_lock", return_value=lock))
        stack.enter_context(patch("fg.data.discover", return_value=({}, [], 0)))
        stack.enter_context(patch("fg.common.note"))
        return root, stack, lock

    def test_failed_preflight_keeps_survey_and_journal_before_run_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, stack, lock = self.configured_runner(temporary, check=False)
            with stack, patch.object(run, "preflight", side_effect=RuntimeError("missing dependency")):
                with self.assertRaisesRegex(RuntimeError, "missing dependency"):
                    run.main()
            self.assertFalse((root / "LATEST.json").exists())
            latest = json.loads((root / "LATEST_VISIT.json").read_text())
            self.assertEqual(latest["status"], "failed")
            journal = Path(latest["journal"])
            self.assertTrue((journal / "survey/summary.json").is_file())
            self.assertTrue(Path(latest["audit"]).is_file())
            self.assertIn("export_review/visit_audit", latest["audit"])
            self.assertNotIn("missing dependency", Path(latest["audit"]).read_text())
            self.assertIn("missing dependency", (journal / "failure.json").read_text())
            self.assertIn("stage_failed", (journal / "events.jsonl").read_text())
            lock.close.assert_not_called()

    def test_survey_only_never_checks_gpu_or_trains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, stack, lock = self.configured_runner(temporary, check=False)
            args = SimpleNamespace(check=False, survey_only=True)
            with stack, patch.object(run, "config_args", return_value=(args, run.config_args()[1])), \
                    patch.object(run, "preflight", side_effect=AssertionError("must not import ML")), \
                    patch.object(run, "run_stage", side_effect=AssertionError("must not train")):
                run.main()
            latest = json.loads((root / "LATEST_VISIT.json").read_text())
            self.assertEqual(latest["status"], "complete")
            self.assertEqual(latest["mode"], "survey_only")
            self.assertTrue(Path(latest["audit"]).is_file())
            self.assertFalse((root / "LATEST.json").exists())
            lock.close.assert_not_called()

    def test_check_preserves_latest_analysis_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, stack, lock = self.configured_runner(temporary, check=True)
            latest = root / "LATEST.json"
            previous = b'{"run":"previous-completed-analysis","report":"previous/report.html"}\n'
            latest.write_bytes(previous)
            with stack, patch.object(run, "run_stage") as stage:
                run.main()
            stage.assert_not_called()
            lock.close.assert_called_once()
            self.assertEqual(latest.read_bytes(), previous)
            self.assertFalse(list(root.glob("mock-*/status.json")))

    def test_failed_stage_details_stay_inside_internal(self):
        secret = "record-PRIVATE-123 at /private/raw/record-PRIVATE-123.json"
        with tempfile.TemporaryDirectory() as temporary:
            root, stack, lock = self.configured_runner(temporary, check=False)
            with stack, patch.object(run, "run_stage", side_effect=ValueError(secret)):
                with self.assertRaisesRegex(ValueError, "record-PRIVATE"):
                    run.main()
            latest = json.loads((root / "LATEST.json").read_text())
            analysis = Path(latest["run"])
            status = json.loads((analysis / "status.json").read_text())
            self.assertEqual(status, {"status": "failed", "type": "ValueError", "details": "internal/failure.json"})
            self.assertNotIn(secret, json.dumps(status))
            failure = json.loads((analysis / status["details"]).read_text())
            self.assertEqual(failure["error"], secret)
            self.assertIn(secret, failure["traceback"])
            self.assertFalse((analysis / "failure.json").exists())
            lock.close.assert_called_once()

    def test_onsite_stage_is_in_review_and_linked_in_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, stack, lock = self.configured_runner(temporary, check=False)
            with stack, patch.object(run, "run_stage") as stage:
                run.main()
            latest = json.loads((root / "LATEST.json").read_text())
            self.assertEqual(Path(latest["onsite_figures"]), Path(latest["export_review"]) / "onsite_figures/index.html")
            names = [call.args[1] for call in stage.call_args_list]
            self.assertEqual(names[-3:], ["report", "export_review", "onsite_figures"])
            self.assertEqual(stage.call_args_list[-1].args[0], Path(latest["run"]))
            self.assertEqual(stage.call_args_list[-1].kwargs["target"], Path(latest["export_review"]) / "onsite_figures")

    def test_nested_guide_stage_resumes_and_detects_tampering_without_changing_export_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "export_review"
            with patch("fg.common.note"):
                run.run_stage(root, "export_review", lambda out: (out / "report.html").write_text("report"))
                original = (root / ".state/export_review.json").read_bytes()
                target = export / "onsite_figures"
                action = Mock(side_effect=lambda out: (out / "index.html").write_text("guide"))
                run.run_stage(root, "onsite_figures", action, target=target)
                run.run_stage(root, "export_review", Mock(side_effect=AssertionError("must resume")))
                run.run_stage(root, "onsite_figures", action, target=target)
                action.assert_called_once()
                self.assertEqual((root / ".state/export_review.json").read_bytes(), original)
                (target / "index.html").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "변경/삭제"):
                    run.run_stage(root, "onsite_figures", action, target=target)


if __name__ == "__main__":
    unittest.main()
