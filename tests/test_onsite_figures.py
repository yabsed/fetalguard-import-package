"""Check interpretation-critical quantities and additive existing-run generation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.common import write_json
from fg.onsite_figures import Guide, difference, operating_counts, reading_rates, native_value, run_onsite_figures, choose_font
from tools.build_onsite_figures import build_existing_run


class OnsiteFigureTests(unittest.TestCase):
    def test_flipped_comparison_reverses_interval_endpoints(self):
        result = {"metrics": {"auprc": {"difference": -.02, "ci95": [-.07, .01]}}}
        self.assertEqual(difference(result, "auprc", reverse=True), (.02, [-.01, .07]))
        self.assertEqual(difference({}, "auprc", reverse=True), (None, None))

    def test_threshold_ties_and_record_denominators(self):
        frame = pd.DataFrame(dict(record_id=["normal", "normal", "quiet", "mixed", "mixed"],
            seg_idx=[0, 1, 0, 0, 1], target=[0, 0, 0, 1, 1], score=[.5, .1, .2, .7, .1],
            threshold=[.5] * 5, record_all_normal=[1, 1, 1, 0, 0]))
        matrix, alarmed, total = operating_counts(frame)
        np.testing.assert_array_equal(matrix, [[2, 1], [1, 1]])
        self.assertEqual((alarmed, total), (1, 2))  # records, not three normal segments
        with self.assertRaisesRegex(ValueError, "unique"):
            operating_counts(pd.concat([frame, frame.iloc[:1]]))

    def test_absent_reading_group_is_not_zero_risk(self):
        result = {"reading_outcome_cells": [dict(reading_positive=1, outcome_positive=0, records=6),
                                           dict(reading_positive=1, outcome_positive=1, records=2)]}
        rates = reading_rates(result)
        self.assertTrue(np.isnan(rates[0][0]))
        self.assertEqual(rates[1], (.25, 2, 8))

    def test_intervals_do_not_require_point_inside_bootstrap_interval(self):
        guide = Guide.__new__(Guide)
        guide.korean = False
        fig, ax = plt.subplots()
        try:
            guide.forest(ax, ["outside", "missing"], [.2, None], [[.3, .4], None], "AP")
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [.3, .4])
            self.assertEqual(len(ax.collections), 1)
            self.assertIn("NA", [t.get_text() for t in ax.texts])
        finally:
            plt.close(fig)

    def test_bounds_readable_and_font_fallback_is_offline(self):
        self.assertEqual(native_value(130.45), "130.4")
        self.assertNotEqual(native_value(131.1), native_value(139.8))
        with patch("fg.onsite_figures.font_manager.fontManager.ttflist", []):
            self.assertEqual(choose_font(), ("DejaVu Sans", False))

    def test_builder_is_additive_atomic_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            internal = root / "internal"
            write_json(internal / "run_manifest.json", {"config": {"profile": "mock"}})
            write_json(root / "status.json", {"status": "complete"})
            write_json(internal / ".state/report.json", {"files": {"report.html": "original-hash"}})
            marker = (internal / ".state/report.json").read_bytes()

            def render(run, out, cfg):
                (out / "index.html").write_text("synthetic guide")

            with patch("tools.build_onsite_figures.run_onsite_figures", side_effect=render):
                index = build_existing_run(root)
                self.assertEqual(index, internal / "onsite_figures/index.html")
                self.assertEqual(index.read_text(), "synthetic guide")
                self.assertEqual((internal / ".state/report.json").read_bytes(), marker)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    build_existing_run(root)
                with self.assertRaisesRegex(ValueError, "directory name"):
                    build_existing_run(root, "../export_review")
            with patch("tools.build_onsite_figures.run_onsite_figures", side_effect=RuntimeError("render failed")):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    build_existing_run(root, "onsite_failed")
            self.assertFalse((internal / "onsite_failed").exists())
            self.assertFalse(list(internal.glob("onsite-build-*")))

    def test_wrong_destination_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            internal = root / "internal"
            write_json(internal / "run_manifest.json", {"config": {}})
            for out in (root / "export_review", internal / "experiment_a"):
                with self.assertRaisesRegex(ValueError, "dedicated directory"):
                    run_onsite_figures(internal, out, {})
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
