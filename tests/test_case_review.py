"""Synthetic-only checks for onsite individual case review."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fg.case_review import select_cases, align_signal_rows, run_case_review
from fg.features import CAT28


def predictions():
    return pd.DataFrame({"record_id": [f"r{i:02d}" for i in range(12)],
        "mother_id": [f"m{i:02d}" for i in range(12)], "seg_idx": [0] * 12,
        "target": [1] * 3 + [0] * 6 + [1] * 3,
        "score": [.95, .9, .85, .05, .1, .15, .9, .8, .7, .1, .2, .3],
        "threshold": [.5] * 12, "model": ["cat28"] * 12})


class CaseReviewTests(unittest.TestCase):
    def test_quadrants_extremes_and_ties_are_deterministic(self):
        frame = predictions()
        frame.loc[0, "score"] = .9
        result = select_cases(frame.sample(frac=1, random_state=31))
        self.assertEqual(result.category.tolist(), ["TP", "TP", "TN", "TN", "FP", "FP", "FN", "FN"])
        self.assertEqual(result.record_id.tolist(), ["r00", "r01", "r03", "r04", "r06", "r07", "r09", "r10"])
        pd.testing.assert_frame_equal(result, select_cases(frame.sample(frac=1, random_state=73)))
        # >= is the fixed alarm rule, including exact threshold ties.
        tied = frame.iloc[[0]].assign(score=.5)
        self.assertEqual(select_cases(tied).category.tolist(), ["TP"])

    def test_missing_categories_are_not_filled_with_other_cases(self):
        frame = predictions().iloc[[0, 1, 2]]
        selected = select_cases(frame)
        self.assertEqual(selected.category.tolist(), ["TP", "TP"])
        self.assertEqual(selected.figure.tolist(), ["case_01.png", "case_02.png"])
        empty = select_cases(frame.iloc[:0])
        self.assertTrue(empty.empty)

    def test_signal_keys_control_alignment_not_prediction_order(self):
        frame = predictions()
        source = frame[["record_id", "mother_id", "seg_idx", "target"]].iloc[::-1].reset_index(drop=True)
        selected = align_signal_rows(select_cases(frame), source)
        self.assertEqual(selected.signal_row_index.tolist(), [11, 10, 8, 7, 5, 4, 2, 1])
        corrupt = source.copy()
        corrupt.loc[corrupt.record_id.eq("r00"), "target"] = 0
        with self.assertRaisesRegex(ValueError, "target.*differ"):
            align_signal_rows(select_cases(frame), corrupt)
        with self.assertRaisesRegex(ValueError, "unique keys"):
            align_signal_rows(select_cases(frame), pd.concat([source, source.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "absent"):
            align_signal_rows(select_cases(frame), source.iloc[:-1])

    def test_internal_report_uses_fixed_paths_and_does_not_select_for_shap(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "internal"
            out = run / "report"
            (run / "experiment_a").mkdir(parents=True)
            (run / "data").mkdir()
            frame = predictions()
            frame.loc[0, "record_id"] = "<script>alert(1)</script>"
            frame.to_csv(run / "experiment_a/test_predictions.csv", index=False)
            source = frame[["record_id", "mother_id", "seg_idx", "target"]].iloc[::-1].reset_index(drop=True)
            for feature_number, feature in enumerate(CAT28):
                source[feature] = np.arange(len(source), dtype=float) * 100 + feature_number + .125
            source.to_csv(run / "data/segments.csv", index=False)
            raw = np.stack([np.full((2, 150), 100 + index, np.float32) for index in range(len(source))])
            np.save(run / "data/signals.npy", raw)
            # Only the third (unselected) FN has SHAP; chosen error cases must not change.
            frame.iloc[[11]][["record_id", "seg_idx", "target"]].assign(
                value_stv=.5, shap_stv=.75).to_csv(run / "experiment_a/local_explanations.csv", index=False)
            self.assertEqual(run_case_review(run, out), "case_review.html")
            result = pd.read_csv(out / "case_review.csv")
            self.assertEqual(result.loc[result.category.eq("FN"), "record_id"].tolist(), ["r09", "r10"])
            self.assertTrue(result.shap_status.str.startswith("unavailable").all())
            feature_columns = ["value_" + feature for feature in CAT28]
            self.assertEqual([column for column in result if column.startswith("value_")], feature_columns)
            self.assertTrue(np.isfinite(result[feature_columns].to_numpy()).all())
            expected = source.set_index(["record_id", "seg_idx"])
            for row in result.itertuples(index=False):
                for feature in CAT28:
                    self.assertEqual(getattr(row, "value_" + feature), expected.loc[(row.record_id, row.seg_idx), feature])
            markup = (out / "case_review.html").read_text()
            self.assertNotIn("<script>alert(1)</script>", markup)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", markup)
            self.assertIn("현장 전용", markup)
            self.assertEqual(markup.count("<h3>Cat28 입력 인자 — 28개 전체</h3>"), len(result))
            for feature in CAT28:
                self.assertEqual(markup.count("<td>" + feature + "</td>"), len(result))
            self.assertEqual({path.name for path in out.iterdir()},
                {"case_review.html", "case_review.csv"} | {f"case_{index:02d}.png" for index in range(1, 9)})
            self.assertTrue(all((out / figure).stat().st_size > 1000 for figure in result.figure))
            # No sibling/export paths are created, even when source IDs contain separators.
            self.assertEqual({path.name for path in Path(temporary).iterdir()}, {"internal"})

    def test_missing_and_nonfinite_source_features_fail_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "internal"
            (run / "experiment_a").mkdir(parents=True)
            (run / "data").mkdir()
            frame = predictions().iloc[:1]
            frame.to_csv(run / "experiment_a/test_predictions.csv", index=False)
            source = frame[["record_id", "mother_id", "seg_idx", "target"]].copy()
            source.to_csv(run / "data/segments.csv", index=False)
            with self.assertRaisesRegex(ValueError, "all 28 Cat28"):
                run_case_review(run, run / "report")
            for feature in CAT28:
                source[feature] = 1.
            source["stv"] = np.nan
            source.to_csv(run / "data/segments.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Cat28 features contain nonfinite"):
                run_case_review(run, run / "report")
            self.assertFalse((run / "report/case_review.html").exists())
            self.assertFalse(list((run / "report").glob("case_*.png")))

    def test_outside_internal_destination_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "inside the internal run"):
                run_case_review(root / "internal", root / "export_review")
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
