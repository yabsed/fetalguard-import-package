"""Adversarial aggregate export tests use synthetic identifiers only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.common import write_json
from fg.export_review import collect_tables, run_export_review, validate_review_bundle, SUPPRESSED


class ExportReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.internal = self.root / "internal"
        self.internal.mkdir()
        self.out = self.root / "export_review"
        self.cfg = {"profile": "mock", "seed": 42, "export_min_mothers": 10, "budget": {"seeds": [42], "bootstrap": 20}}
        self.secret = "SECRET_PATIENT_ID_AND_PATH"
        frame = pd.DataFrame({"mother_id": [self.secret + str(i) for i in range(40)],
                              "record_id": [self.secret + "record" + str(i) for i in range(40)],
                              "seg_idx": 0, "target": [0, 1] * 20, "model": "cat28", "score": [.2, .8] * 20,
                              "threshold": .5, "site": self.secret + "hospital", "record_all_normal": [1, 0] * 20})
        self.save_csv("experiment_a/test_predictions.csv", frame)
        self.save_csv("data/segments.csv", frame.drop(columns=["model", "score", "threshold"]))
        self.save_csv("splits/segments.csv", frame[["record_id", "mother_id", "seg_idx"]].assign(split="test"))
        self.result = {"status": "ok", "n": 40, "mothers": 40, "positive_mothers": 20, "negative_mothers": 20,
                       "point": {"auroc": .8, "auprc": .7, "specificity": .9},
                       "ci95": {"auroc": [.7, .9], "auprc": [.6, .8]}, "untrusted": self.secret}
        write_json(self.internal / "experiment_a/holdout_metrics.json", {"cat28": self.result, self.secret: self.result})
        write_json(self.internal / "run_manifest.json", {"package_sha256": "a" * 64, "data_root": self.secret,
                    "config": {"output_root": self.secret}, "inputs": [{"record_id": self.secret}]})

    def save_csv(self, relative, frame):
        path = self.internal / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def test_arbitrary_nested_fields_identifiers_paths_and_unknown_files_do_not_escape(self):
        write_json(self.internal / "experiment_a/model.json", {"weights": self.secret})
        write_json(self.internal / "experiment_a/paired_cat28_minus_comparator.json", {
            "cat18": {"metrics": {"auroc": {"difference": .02, "ci95": [-.01, .05], "patient": self.secret}}, "path": self.secret}})
        write_json(self.internal / "supplementary/loso_metrics.json", {self.secret: {"raw": self.result, "platt": self.result}})
        frame = pd.read_csv(self.internal / "experiment_a/test_predictions.csv").assign(arm="raw", site=self.secret)
        self.save_csv("supplementary/loso_predictions.csv", frame)
        run_export_review(self.internal, self.out, self.cfg)
        for path in self.out.rglob("*"):
            if path.suffix in (".csv", ".json", ".md", ".html"):
                text = path.read_text()
                self.assertNotIn(self.secret, text, path)
                self.assertNotIn(str(self.internal), text, path)
        self.assertFalse((self.out / "model.json").exists())
        self.assertEqual(pd.read_csv(self.out / "csv/loso.csv").site.tolist(), ["S01", "S01"])
        self.assertFalse(list(self.out.glob("*.csv")))
        self.assertIn("href='csv/loso.csv'", (self.out / "report.html").read_text())
        summary = validate_review_bundle(self.out)
        self.assertTrue(summary["manifest_hashes_match"])
        manifest = json.loads((self.out / "EXPORT_MANIFEST.json").read_text())
        self.assertEqual(manifest["review_status"], "pending_institution_review")
        self.assertFalse(manifest["screening_is_approval"])

    def test_small_positive_mother_cohort_suppresses_counts_scores_and_intervals(self):
        frame = pd.read_csv(self.internal / "experiment_a/test_predictions.csv").iloc[:12].copy()
        frame["target"] = [1] + [0] * 11
        self.save_csv("experiment_a/test_predictions.csv", frame)
        rows = collect_tables(self.internal, self.cfg)["model_comparison"]
        self.assertEqual(rows.iloc[0].review_status, SUPPRESSED)
        for key in ("n", "mothers", "positive_mothers", "auroc", "auroc_lo", "auprc"):
            self.assertTrue(pd.isna(rows.iloc[0][key]), key)

    def test_unknown_denominators_fail_closed(self):
        write_json(self.internal / "official/yolo_metrics.json", {"metrics": {"point": {"auroc": .99}, "n": 100}})
        row = collect_tables(self.internal, self.cfg)["official_reference"].iloc[0]
        self.assertEqual(row.review_status, SUPPRESSED)
        self.assertTrue(pd.isna(row.auroc))

    def test_cv_and_seed_alarm_aggregates_need_their_own_mother_denominators(self):
        frame = pd.read_csv(self.internal / "data/segments.csv")
        dev = frame.copy()
        dev["record_id"] = "DEV_" + dev.record_id
        dev["mother_id"] = "DEV_" + dev.mother_id
        self.save_csv("data/segments.csv", pd.concat([frame, dev]))
        self.save_csv("splits/segments.csv", pd.concat([
            frame[["record_id", "mother_id", "seg_idx"]].assign(split="test"),
            dev[["record_id", "mother_id", "seg_idx"]].assign(split="train")]))
        source = pd.DataFrame([{"model": "cat28", "seed": 42, "auroc": .8,
                                "normal_record_alarm_rate": .025, "false_alarms_per_normal_hour": .3}])
        for name in ("cv_summary", "seed_holdout_metrics"):
            self.save_csv("experiment_a/" + name + ".csv", source)
        exported = collect_tables(self.internal, self.cfg)
        onsite = collect_tables(self.internal, self.cfg, screened=False)
        for name in ("cv_summary", "seed_holdout_metrics"):
            with self.subTest(table=name):
                row = exported[name].iloc[0]
                self.assertEqual(row.auroc, .8)
                self.assertTrue(pd.isna(row.normal_record_alarm_rate))
                self.assertTrue(pd.isna(row.false_alarms_per_normal_hour))
                self.assertEqual(onsite[name].iloc[0].normal_record_alarm_rate, .025)
                self.assertEqual(onsite[name].iloc[0].false_alarms_per_normal_hour, .3)

    def test_a_small_training_bin_suppresses_entire_feature_distribution(self):
        self.save_csv("experiment_a/feature_response_train.csv", pd.DataFrame([
            {"feature": "stv", "bin": 0, "n_mothers": 40, "positive_mothers": 20, "negative_mothers": 20, "positive_fraction": .5},
            {"feature": "stv", "bin": 1, "n_mothers": 12, "positive_mothers": 1, "negative_mothers": 11, "positive_fraction": .083}]))
        rows = collect_tables(self.internal, self.cfg)["feature_response_train"]
        self.assertTrue((rows.review_status == SUPPRESSED).all())
        self.assertTrue(rows.positive_fraction.isna().all())
        self.assertTrue(rows.n_mothers.isna().all())

    def test_shap_uses_actual_contributors_not_the_large_test_cohort(self):
        self.save_csv("experiment_a/feature_importance.csv", pd.DataFrame([{"feature": "stv", "mean_abs_shap": 1.5}]))
        source = pd.read_csv(self.internal / "experiment_a/test_predictions.csv")
        self.save_csv("experiment_a/local_explanations.csv", source[["record_id", "seg_idx", "target"]].iloc[:1])
        row = collect_tables(self.internal, self.cfg)["feature_importance"].iloc[0]
        self.assertEqual(row.review_status, SUPPRESSED)
        self.assertTrue(pd.isna(row.mean_abs_shap))
        self.assertTrue(pd.isna(row.contributing_mothers))
        self.save_csv("experiment_a/local_explanations.csv", source[["record_id", "seg_idx", "target"]])
        row = collect_tables(self.internal, self.cfg)["feature_importance"].iloc[0]
        self.assertEqual(row.mean_abs_shap, 1.5)
        self.assertEqual(row.contributing_mothers, 40)

    def test_inference_cost_separates_extraction_and_normalization_scope(self):
        write_json(self.internal / "data/summary.json", {"feature_extraction_ms_per_segment": {"smooth30": 1.2, "smooth0": .9}})
        self.result.update(inference_timing={"median_batch_seconds": .04, "segments": 40, "median_microseconds_per_segment": 1000, "repeats": 3}, model_bytes=1234)
        write_json(self.internal / "experiment_a/holdout_metrics.json", {"cat28": self.result})
        self.save_csv("experiment_b/inference_cost.csv", pd.DataFrame([{"model": "cnn_small_channel_maxabs", "milliseconds_per_window": 2.5}]))
        rows = collect_tables(self.internal, self.cfg)["inference_cost"].set_index("model")
        self.assertEqual(rows.loc["cat28", "feature_extraction_smooth30_ms_per_segment"], 1.2)
        self.assertEqual(rows.loc["cat28", "feature_extraction_smooth0_ms_per_segment"], .9)
        self.assertEqual(rows.loc["cnn_small_channel_maxabs", "scope"], "raw_window_normalization_and_inference_excludes_loading_interpolation")

    def test_emr_selection_exports_only_screened_seed_and_validation_fields(self):
        frame = pd.read_csv(self.internal / "data/segments.csv")
        valid = frame.copy()
        valid["record_id"] = "VAL_" + valid.record_id
        valid["mother_id"] = "VAL_" + valid.mother_id
        self.save_csv("data/segments.csv", pd.concat([frame, valid]))
        self.save_csv("splits/segments.csv", pd.concat([frame[["record_id", "mother_id", "seg_idx"]].assign(split="test"),
                                                      valid[["record_id", "mother_id", "seg_idx"]].assign(split="val")]))
        write_json(self.internal / "supplementary/emr_added.json", {"selection": {
            "with_emr": {"seed": 43, "validation_auprc": .6, "threshold": .4, "model_file": self.secret},
            "cat28": {"seed": 44, "validation_auprc": .7, "threshold": .5}}})
        self.save_csv("supplementary/emr_validation_selection.csv", pd.DataFrame([{"arm": "cat28", "seed": 42, "validation_auprc": .5}]))
        rows = collect_tables(self.internal, self.cfg)["emr_selection"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows.loc[(rows.arm == "with_emr") & (rows.selection_stage == "selected"), "seed"].iloc[0], 43)
        self.assertNotIn(self.secret, rows.to_csv(index=False))
        self.assertNotIn("model_file", rows.columns)
        strict = collect_tables(self.internal, dict(self.cfg, export_min_mothers=21))["emr_selection"]
        self.assertTrue(strict.seed.isna().all())

    def test_outcome_descriptors_screen_the_actual_observed_subgroup(self):
        cohort = {"records": 40, "mothers": 40, "reading_positive_mothers": 20, "reading_negative_mothers": 20,
                  "maternal_age": {"observed_records": 25, "observed_mothers": 25, "missing_records": 15, "missing_mothers": 15, "mean": 32, "median": 31},
                  "gestational_age": {"observed_records": 1, "observed_mothers": 1, "missing_records": 39, "missing_mothers": 39, "mean": 34, "median": 34}}
        write_json(self.internal / "supplementary/outcomes.json", {"ph_lt_7_20": {"cohorts": {"observed_valid": cohort}}})
        rows = collect_tables(self.internal, self.cfg)["outcome_cohort_descriptors"]
        age = rows.loc[(rows.cohort == "observed_valid") & (rows.feature == "maternal_age")].iloc[0]
        ga = rows.loc[(rows.cohort == "observed_valid") & (rows.feature == "gestational_age")].iloc[0]
        self.assertEqual(age["mean"], 32)
        self.assertEqual(age["median"], 31)
        self.assertEqual(ga.review_status, SUPPRESSED)
        self.assertTrue(pd.isna(ga["mean"]))
        self.assertTrue(pd.isna(ga["observed_mothers"]))

    def test_mock_evidence_never_claims_a_research_result(self):
        write_json(self.internal / "experiment_a/paired_cat28_minus_comparator.json", {"best_single_nonlinear": {
            "metrics": {"auroc": {"difference": .15, "ci95": [.1, .2]}}}})
        rows = collect_tables(self.internal, self.cfg)["hypothesis_evidence"]
        self.assertTrue((rows.evidence == "execution_test_only").all())
        full = collect_tables(self.internal, dict(self.cfg, profile="full"))["hypothesis_evidence"]
        self.assertEqual(full.loc[full.question == "H2", "evidence"].iloc[0], "positive_interval")

    def test_numeric_schema_does_not_export_injected_strings(self):
        self.result["point"]["auroc"] = self.secret
        self.result["ci95"]["auroc"] = [self.secret, {"patient": self.secret}]
        write_json(self.internal / "experiment_a/holdout_metrics.json", {"cat28": self.result})
        rows = collect_tables(self.internal, self.cfg)["model_comparison"]
        self.assertTrue(pd.isna(rows.iloc[0].auroc))
        self.assertTrue(pd.isna(rows.iloc[0].auroc_lo))

    def test_symlink_source_rejected(self):
        target = self.internal / "experiment_a/holdout_metrics.json"
        target.unlink()
        outside = self.root / "outside.json"
        write_json(outside, {"cat28": self.result})
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            collect_tables(self.internal, self.cfg)

    def test_existing_complete_bundle_is_resumable_but_extra_file_is_rejected(self):
        run_export_review(self.internal, self.out, self.cfg)
        run_export_review(self.internal, self.out, self.cfg)
        (self.out / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            run_export_review(self.internal, self.out, self.cfg)

    def test_failed_build_publishes_no_partial_bundle(self):
        self.out.mkdir()
        with patch("fg.export_review.write_review_report", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_export_review(self.internal, self.out, self.cfg)
        self.assertEqual(list(self.out.iterdir()), [])
        run_export_review(self.internal, self.out, self.cfg)
        validate_review_bundle(self.out)

    def test_manifest_hash_detects_modified_data(self):
        run_export_review(self.internal, self.out, self.cfg)
        (self.out / "csv/model_comparison.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_review_bundle(self.out)

    def test_onsite_extension_has_separate_inventory_and_cannot_weaken_image_screening(self):
        run_export_review(self.internal, self.out, self.cfg)
        manifest_file = self.out / "EXPORT_MANIFEST.json"
        original = manifest_file.read_bytes()
        from fg.export_diagnostics import build_export_diagnostics
        build_export_diagnostics(self.internal, self.out / "visit_audit", self.cfg, images=False)
        self.assertIn("visit_audit", validate_review_bundle(self.out)["screened_diagnostics"])
        guide = self.out / "onsite_figures"
        guide.mkdir()
        (guide / "index.html").write_text(self.secret)
        (guide / "README.md").write_text("Onsite only")
        write_json(guide / "manifest.json", {"purpose": "onsite_understanding", "source_files": {},
            "review_status": "onsite_only_not_screened",
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in guide.iterdir()}})
        result = validate_review_bundle(self.out)
        self.assertEqual(result["onsite_only"]["onsite_figures"]["status"], "onsite_only_not_screened")
        self.assertEqual(manifest_file.read_bytes(), original)
        (guide / "private.csv").write_text(self.secret)
        with self.assertRaisesRegex(ValueError, "Unexpected file"):
            validate_review_bundle(self.out)
        (guide / "private.csv").unlink()
        (guide / "index.html").write_text("modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_review_bundle(self.out)

    def test_even_rehashed_html_cannot_link_to_internal_data(self):
        run_export_review(self.internal, self.out, self.cfg)
        path = self.out / "report.html"
        path.write_text(path.read_text() + "<a href='../internal/data/records.csv'>private</a>")
        manifest_file = self.out / "EXPORT_MANIFEST.json"
        manifest = json.loads(manifest_file.read_text())
        entry = next(item for item in manifest["files"] if item["file"] == "report.html")
        entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        write_json(manifest_file, manifest)
        with self.assertRaisesRegex(ValueError, "Unexpected link"):
            validate_review_bundle(self.out)


if __name__ == "__main__":
    unittest.main()
