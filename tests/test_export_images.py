"""Synthetic-only checks of image coverage, plotted values and submission isolation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, PngImagePlugin

from fg.common import write_json
from fg.export_images import (SECTIONS, _panel, write_image_atlas,
                              validate_image_directory, ROWS_PER_PAGE)
from fg.export_review import SCREENED, SUPPRESSED, EXPORT_TABLES
from tools.build_image_review import build_existing_run


def synthetic_tables():
    """Every fixed schema, with synthetic aggregate values, for renderer coverage."""
    label_values = {"model": "cat28", "left": "cat28", "right": "cat18", "scope": "shared_holdout",
                    "feature": "stv", "metric": "auroc", "split": "test", "comparator": "cat18",
                    "family": "linear", "measurement_status": "ok", "label": "BaseLine",
                    "normalization": "channel_maxabs", "arm": "clinical", "outcome": "ph_lt_7_20",
                    "cohort": "observed_valid", "selection_stage": "selected", "device": "cpu",
                    "site": "S01", "question": "H2", "contrast": "cat28_minus_best_single_nonlinear",
                    "evidence": "execution_test_only", "primary_metric": "auroc",
                    "left_normalization": "channel_maxabs", "right_normalization": "per_segment_z",
                    "seed": 42, "width": 8, "reading_positive": 1, "outcome_positive": 0}
    tables = {}
    for name, section in SECTIONS.items():
        row = {key: label_values[key] for key in set(section.labels + section.groups)}
        row.update({key: .75 for key in section.metrics})
        row.update(review_status=SCREENED, difference=.1, ci95_low=-.02, ci95_high=.2)
        tables[name] = pd.DataFrame([row])
    tables["feature_response_train"] = pd.DataFrame([
        dict(feature=feature, bin=i, positive_fraction=.2 + i * .1, n_mothers=60, n_segments=120,
             review_status=SCREENED)
        for feature in ("stv", "fhr_min", "figo_baseline_var", "n_decel", "fhr_mean") for i in range(4)])
    return tables


class ExportImageTests(unittest.TestCase):
    def test_ci_endpoints_and_missing_values_are_not_replaced_with_zero(self):
        rows = [{"review_status": SCREENED, "auroc": .8, "auroc_lo": .7, "auroc_hi": .9},
                {"review_status": SUPPRESSED, "auroc": 987654321, "auroc_lo": 987654320},
                {"review_status": SCREENED, "auroc": None},
                {"review_status": SCREENED, "auroc": 0}]
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        _panel(ax, rows, "auroc", ["A", "B", "C", "D"])
        texts = [t.get_text() for t in ax.texts]
        self.assertIn("0.8000\n[0.7000, 0.9000]", texts)
        self.assertIn("SUPPRESSED", texts)
        self.assertIn("NA", texts)
        self.assertIn("0", texts)
        points = [list(line.get_xdata()) for line in ax.lines]
        self.assertEqual(points, [[.7, .9], [.7, .9], [.8], [0]])
        self.assertNotIn("987654321", " ".join(texts))

    def test_all_fixed_sections_and_all_response_features_are_rendered(self):
        saved = []
        def capture(fig, path, pyplot):
            saved.append((path.name, [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()] +
                          [ax.get_title(loc="left") for ax in fig.axes]))
            pyplot.close(fig)
            return path
        self.assertEqual(set(SECTIONS), EXPORT_TABLES)
        with tempfile.TemporaryDirectory() as temporary, patch("fg.export_images._save", side_effect=capture), patch("fg.export_images.validate_image_directory"):
            write_image_atlas(Path(temporary), synthetic_tables(), {"profile": "mock"})
        for name in SECTIONS:
            self.assertTrue(any("_" + name + "_p" in n for n, _ in saved), name)
        responses = [n for n, _ in saved if "feature_response_train" in n]
        self.assertEqual(len(responses), 2)
        self.assertTrue(any("fhr mean" in labels for _, labels in saved))

    def test_long_candidate_list_is_paginated_without_dropping_rows(self):
        seen = []
        def capture(fig, path, pyplot):
            if "validation_selection" in path.name:
                seen.extend(t.get_text() for ax in fig.axes for t in ax.get_yticklabels())
            pyplot.close(fig)
            return path
        frame = pd.DataFrame([dict(model="cat28", seed=i, validation_auprc=.7, review_status=SCREENED)
                              for i in range(ROWS_PER_PAGE * 2 + 1)])
        with tempfile.TemporaryDirectory() as temporary, patch("fg.export_images._save", side_effect=capture), patch("fg.export_images.validate_image_directory"):
            files = write_image_atlas(Path(temporary), {"validation_selection": frame}, {})
        self.assertEqual(len([p for p in files if "validation_selection" in p.name]), 3)
        self.assertEqual(len(seen), len(frame))
        self.assertIn("seed=12", seen[-1])

    def test_onsite_and_statusless_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for row in ({"model": "cat28", "auroc": .8},
                        {"model": "cat28", "auroc": .8, "review_status": "onsite_only_not_screened"}):
                with self.assertRaisesRegex(ValueError, "screen"):
                    write_image_atlas(Path(temporary), {"model_comparison": pd.DataFrame([row])}, {})

    def test_png_directory_rejects_csv_metadata_and_trailing_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = root / "00_coverage_p001.png"
            image = Image.new("RGB", (1200, 1200), "white")
            image.save(png)
            self.assertEqual(validate_image_directory(root)["png_files"], 1)
            (root / "values.csv").write_text("private")
            with self.assertRaisesRegex(ValueError, "Unexpected file"):
                validate_image_directory(root)
            (root / "values.csv").unlink()
            info = PngImagePlugin.PngInfo()
            info.add_text("patient", "SECRET_PATIENT")
            image.save(png, pnginfo=info)
            with self.assertRaisesRegex(ValueError, "metadata"):
                validate_image_directory(root)
            image.save(png)
            with png.open("ab") as stream:
                stream.write(b"SECRET_PATIENT")
            with self.assertRaisesRegex(ValueError, "trailing"):
                validate_image_directory(root)

    def test_existing_run_builder_preserves_original_budget_and_can_only_raise_screen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            internal = root / "internal"
            config = {"profile": "mock", "export_min_mothers": 10, "budget": {"bootstrap": 50}}
            write_json(internal / "run_manifest.json", {"config": config})
            with patch("tools.build_image_review.run_export_review") as build, patch("tools.build_image_review.validate_review_bundle"):
                output = build_existing_run(root, minimum=20)
            self.assertEqual(output, root / "image_review/images")
            used = build.call_args.args[2]
            self.assertEqual(used["export_min_mothers"], 20)
            self.assertEqual(used["budget"], {"bootstrap": 50})
            with self.assertRaisesRegex(ValueError, "retain or raise"):
                build_existing_run(root, minimum=2)
            with self.assertRaisesRegex(ValueError, "not overwritten"):
                build_existing_run(root, root)
            with self.assertRaisesRegex(ValueError, "outside internal"):
                build_existing_run(root, internal / "new")


if __name__ == "__main__":
    unittest.main()
