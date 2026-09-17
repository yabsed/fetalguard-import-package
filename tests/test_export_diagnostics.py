"""Synthetic privacy and integration checks for review diagnostics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.export_diagnostics import build_export_diagnostics, validate_diagnostics, Reader, collect, SUPPRESSED
from fg.telemetry import save_json
from fg.survey import write_csv


class DiagnosticReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "internal"
        self.out = self.root / "export_review/visit_audit"
        self.secret = "PRIVATE_MOTHER_SITE_AND_PATH"
        self.cfg = dict(profile="mock", seed=42, export_min_mothers=10,
                        budget=dict(seeds=[42], cv_folds=2, tree_iterations=100, tree_patience=10))
        save_json(self.run / "run_manifest.json", dict(config=self.cfg))
        records, segments, pred = [], [], []
        for i in range(240):
            rid, mid = self.secret + str(i), self.secret + "m" + str(i)
            split = ("train", "val", "test")[i // 80]
            records.append(dict(record_id=rid, mother_id=mid, site=self.secret, n_segments=1, n_kept=1,
                                any_abnormal=i % 2, is_twin=False, gestational_age=38, **{"emr_Mother.Height": 160}))
            row = dict(record_id=rid, mother_id=mid, site=self.secret, seg_idx=0, target=i % 2, split=split)
            segments.append(row)
            if split == "test":
                pred.append(dict(row, model="cat28", score=.2 if i % 4 < 2 else .8, threshold=.5))
        self.csv("data/records.csv", records)
        self.csv("data/segments.csv", segments)
        self.csv("splits/segments.csv", segments)
        self.csv("experiment_a/test_predictions.csv", pred)
        job = self.run / "experiment_a/models/holdout_cat28_42.cbm"
        job.parent.mkdir()
        job.write_bytes(b"synthetic model never read")
        save_json(job.with_name(job.name + ".training.json"), dict(family="cat", seed=42,
            iterations_run=100, best_iteration=99, cap=100, patience=10, score_column="validation/PRAUC",
            score_metric=self.secret, stop_reason=self.secret))
        write_csv(job.with_name(job.name + ".history.csv"), [{"validation/PRAUC": .6, "learn/Logloss": .4, "private": self.secret}],
                  ["validation/PRAUC", "learn/Logloss", "private"])
        save_json(self.run / ".state/data.json", dict(seconds=1))

    def csv(self, name, rows):
        path = self.run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_all_review_text_is_self_contained_and_sources_are_preserved(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.assertTrue(build_export_diagnostics(self.run, self.out, images=False).is_file())
        for p in self.out.rglob("*"):
            if p.is_file():
                text = p.read_text()
                self.assertNotIn(self.secret, text, p)
                self.assertNotIn(str(self.run), text, p)
        confusion = pd.read_csv(self.out / "csv/confusion.csv")
        self.assertEqual(confusion.segments.sum(), 80)
        self.assertEqual(confusion.mothers.tolist(), [20] * 4)
        training = pd.read_csv(self.out / "csv/training_diagnostics.csv")
        self.assertEqual(training.iloc[0].model, "cat28")
        self.assertEqual(training.iloc[0].stop_reason, "budget_cap")
        history = pd.read_csv(self.out / "csv/training_history.csv")
        self.assertEqual(history.iloc[0].validation_score, .6)
        self.assertTrue(validate_diagnostics(self.out)["files"] > 20)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_rare_mother_cells_suppress_entire_confusion_partition(self):
        pred = pd.read_csv(self.run / "experiment_a/test_predictions.csv", dtype={"mother_id": str})
        # 20 rows from the same mother are not 20 independent people.
        pred.loc[(pred.target == 1) & (pred.score > .5), "mother_id"] = "one_private_mother"
        self.csv("experiment_a/test_predictions.csv", pred)
        tables, _ = collect(self.run, None, self.cfg, Reader(self.out))
        self.assertTrue(all(r["review_status"] == SUPPRESSED for r in tables["confusion"]))
        self.assertTrue(all(r["segments"] is None for r in tables["confusion"]))

    def test_unknown_jobs_and_unverified_histories_do_not_publish_scores_or_paths(self):
        job = self.run / "experiment_a/models" / (self.secret + ".cbm")
        job.write_bytes(b"not loaded")
        save_json(job.with_name(job.name + ".training.json"), dict(iterations_run=2, best_iteration=1,
            cap=10, patience=3, family=self.secret, score_column="validation/PRAUC"))
        write_csv(job.with_name(job.name + ".history.csv"), [{"validation/PRAUC": .99}], ["validation/PRAUC"])
        tables, _ = collect(self.run, None, self.cfg, Reader(self.out))
        rows = [r for r in tables["training_history"] if r["model"] == "unclassified"]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["validation_score"])
        self.assertNotIn(self.secret, json.dumps(tables))

    def test_atomic_failure_and_rehashed_external_link_rejection(self):
        with patch("fg.export_diagnostics.render", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                build_export_diagnostics(self.run, self.out, images=False)
        self.assertFalse(self.out.exists())
        build_export_diagnostics(self.run, self.out, images=False)
        page = self.out / "index.html"
        page.write_text("<a href='../../internal/data/records.csv'>private</a>")
        manifest = json.loads((self.out / "EXPORT_MANIFEST.json").read_text())
        manifest["files"]["index.html"] = hashlib.sha256(page.read_bytes()).hexdigest()
        save_json(self.out / "EXPORT_MANIFEST.json", manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate_diagnostics(self.out)

    def test_wrong_destination_symlinks_and_extra_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "export_review"):
            build_export_diagnostics(self.run, self.run / "new_export", images=False)
        build_export_diagnostics(self.run, self.out, images=False)
        (self.out / "private.csv").write_text(self.secret)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            validate_diagnostics(self.out)
        (self.out / "private.csv").unlink()
        (self.out / "external").symlink_to(self.run, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            validate_diagnostics(self.out)

    def test_images_render_from_projection_only(self):
        from fg.export_diagnostics import render_images
        from PIL import Image
        out = self.root / "images"
        out.mkdir()
        tables = {"resources": [dict(metric="process_rss_bytes", mean=100, maximum=200, review_status="screened_aggregate_pending_review")],
                  "training_history": []}
        names = render_images(out, tables)
        self.assertEqual(len(names), 1)
        with Image.open(out / names[0]) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertFalse(image.info)

    def test_diagnostic_only_directory_can_later_receive_primary_bundle(self):
        from fg.export_review import run_export_review
        build_export_diagnostics(self.run, self.out, images=False)
        original = (self.out / "EXPORT_MANIFEST.json").read_bytes()
        def small_primary(run, out, cfg):
            out.mkdir()
            save_json(out / "EXPORT_MANIFEST.json", {"synthetic": True})
        with patch("fg.export_review._build_export_review", side_effect=small_primary), \
                patch("fg.export_review.validate_review_bundle"):
            run_export_review(self.run, self.out.parent, self.cfg)
        self.assertEqual((self.out / "EXPORT_MANIFEST.json").read_bytes(), original)
        self.assertTrue((self.out.parent / "EXPORT_MANIFEST.json").is_file())


if __name__ == "__main__":
    unittest.main()
