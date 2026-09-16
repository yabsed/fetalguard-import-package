"""Deterministic individual-case review, strictly inside the onsite run.

This module never fits models, changes thresholds, or selects cases based on
SHAP availability. Images/identifiers are private artifacts, not export inputs.
"""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import table
from .features import CAT28


KEYS = ["record_id", "seg_idx"]
CATEGORIES = ("TP", "TN", "FP", "FN")
CATEGORY_NAMES = {"TP": "정탐", "TN": "정상 정분류", "FP": "오탐", "FN": "미탐"}


def classify_cases(predictions):
    """Keep Cat28's fixed validation threshold, including conservative ties."""
    required = set(KEYS + ["target", "score", "threshold"])
    if not required.issubset(predictions):
        raise ValueError("Case review needs identifiers, target, score and validation threshold")
    frame = predictions.loc[predictions.model.eq("cat28")].copy() if "model" in predictions else predictions.copy()
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate Cat28 case-review prediction keys")
    if not frame.target.isin([0, 1]).all():
        raise ValueError("Case-review targets must be binary")
    if not np.isfinite(frame[["score", "threshold"]].to_numpy(float)).all() or not frame.score.between(0, 1).all():
        raise ValueError("Case-review scores/thresholds must be finite probabilities")
    frame["record_id"] = frame.record_id.astype(str)
    frame["predicted_label"] = frame.score.ge(frame.threshold).astype(int)
    frame["category"] = np.where(frame.target.eq(1),
        np.where(frame.predicted_label.eq(1), "TP", "FN"),
        np.where(frame.predicted_label.eq(1), "FP", "TN"))
    return frame


def select_cases(predictions, per_category=2):
    if not isinstance(per_category, int) or per_category < 1:
        raise ValueError("Case count must be a positive integer")
    frame = classify_cases(predictions)
    subsets = []
    for category in CATEGORIES:
        # Review high-confidence false alarms and low-score missed positives.
        ascending_score = category in {"FN", "TN"}
        chosen = frame.loc[frame.category.eq(category)].sort_values(
            ["score", "record_id", "seg_idx"], ascending=[ascending_score, True, True], kind="stable").head(per_category)
        subsets.append(chosen)
    selected = pd.concat(subsets, ignore_index=True)
    selected["case_number"] = np.arange(1, len(selected) + 1)
    selected["figure"] = [f"case_{number:02d}.png" for number in selected.case_number]
    return selected


def align_signal_rows(selected, segments):
    """The original segment CSV order is the signals.npy row contract."""
    required = set(KEYS + ["target"])
    if not required.issubset(segments):
        raise ValueError("Signal alignment needs source segment keys and targets")
    if segments.duplicated(KEYS).any() or selected.duplicated(KEYS).any():
        raise ValueError("Case signal alignment requires unique keys")
    source = segments.reset_index(drop=True).copy()
    source["signal_row_index"] = np.arange(len(source))
    columns = KEYS + ["target", "signal_row_index"]
    if "mother_id" in selected and "mother_id" in source:
        columns.append("mother_id")
    joined = selected.merge(source[columns], on=KEYS, how="left", sort=False,
                            validate="one_to_one", suffixes=("", "_source"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Selected review case is absent from source signal rows")
    if not joined.target.eq(joined.target_source).all():
        raise ValueError("Case-review target and source signal target differ")
    if "mother_id_source" in joined and not joined.mother_id.eq(joined.mother_id_source).all():
        raise ValueError("Case-review mother and source signal mother differ")
    joined["signal_row_index"] = joined.signal_row_index.astype(int)
    return joined.drop(columns=["_merge", "target_source", "mother_id_source"], errors="ignore")


def explanation_for_case(row, explanations):
    if explanations.empty:
        return "unavailable_not_in_precomputed_shap_sample", []
    match = explanations.loc[explanations.record_id.eq(row.record_id) & explanations.seg_idx.eq(row.seg_idx)]
    if match.empty:
        return "unavailable_not_in_precomputed_shap_sample", []
    item = match.iloc[0]
    if "target" in item and item.target != row.target:
        raise ValueError("Case-review SHAP target mismatch")
    values = []
    for feature in CAT28:
        contribution = item.get("shap_" + feature, np.nan)
        if not isinstance(contribution, (int, float, np.number)) or not np.isfinite(contribution):
            continue
        value = item.get("value_" + feature, np.nan)
        values.append(dict(feature=feature,
                           value=float(value) if isinstance(value, (int, float, np.number)) and np.isfinite(value) else None,
                           shap_logit=float(contribution)))
    values.sort(key=lambda value: (-abs(value["shap_logit"]), value["feature"]))
    return ("available_precomputed" if values else "unavailable_no_finite_contributions"), values[:5]


def run_case_review(run, out):
    """Return the relative HTML path; all outputs must remain under internal run."""
    run, out = Path(run), Path(out)
    if not out.resolve().is_relative_to(run.resolve()) or out.is_symlink():
        raise ValueError("Individual case review must stay inside the internal run")
    out.mkdir(parents=True, exist_ok=True)
    predictions = table(run / "experiment_a/test_predictions.csv")
    cases = classify_cases(predictions)
    selected = select_cases(predictions)
    segments = table(run / "data/segments.csv")
    selected = align_signal_rows(selected, segments)
    if not set(CAT28).issubset(segments):
        raise ValueError("Case review requires all 28 Cat28 source feature columns")
    features = segments[CAT28].to_numpy(float)
    if not np.isfinite(features[selected.signal_row_index.to_numpy()]).all():
        raise ValueError("Selected case Cat28 features contain nonfinite values")
    signals = np.load(run / "data/signals.npy", mmap_mode="r", allow_pickle=False)
    if signals.shape != (len(segments), 2, 150):
        raise ValueError("Case-review signal array/segment shape mismatch")
    explanation_path = run / "experiment_a/local_explanations.csv"
    explanations = table(explanation_path) if explanation_path.is_file() else pd.DataFrame()
    if not explanations.empty and (not set(KEYS).issubset(explanations) or explanations.duplicated(KEYS).any()):
        raise ValueError("Case-review SHAP rows must have unique segment keys")
    markup = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>현장 전용 사례 검토</title>",
              "<style>body{font:15px system-ui;max-width:1000px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{max-width:100%}.private{color:#a22}section{margin:2em 0;border-top:1px solid #aaa}</style>",
              "<h1 class='private'>현장 전용 — 개별 사례·파형 검토</h1>",
              "<p class='private'>기록 식별자, 개별 예측과 파형을 포함합니다. 이 화면·CSV·이미지는 반출 심사용 묶음에 포함되지 않습니다.</p>",
              "<p>Cat28 테스트 예측에서 TP/TN/FP/FN별 최대 2개를 고릅니다. TP·FP는 점수 내림차순, TN·FN은 오름차순이며 동점은 기록 ID·구간 번호 순입니다. 검증셋으로 정한 임계값을 그대로 사용합니다. 사례를 보고 모델·임계값을 다시 조정하는 용도가 아닙니다.</p>",
              "<p>모든 테스트 사례가 선택 대상입니다. SHAP 계산 여부는 선택에 영향을 주지 않습니다. 선택한 극단 사례는 전체 오류 분포를 대표하는 무작위 표본이 아닙니다. 각 사례의 Cat28 입력 인자 28개는 SHAP 유무와 관계없이 표시됩니다.</p>",
              "<p>파형은 저장된 정규화 전 입력입니다: FHR 0 결측은 보간되어 있고 TOCO는 보존됩니다. Cat28의 특성은 별도 30초 평활 전처리를 거칩니다. SHAP은 모델 점수에 대한 로짓 척도의 기여이며 전문의의 실제 판단 근거나 인과적 원인을 입증하지 않습니다.</p>"]
    availability = pd.DataFrame([dict(category=category, available=int(cases.category.eq(category).sum()),
                                      selected=int(selected.category.eq(category).sum())) for category in CATEGORIES])
    markup.append(availability.to_html(index=False))
    rows = []
    for row in selected.itertuples(index=False):
        signal = signals[row.signal_row_index]
        if not np.isfinite(signal).all():
            raise ValueError("Selected case signal contains nonfinite values")
        factor_values = dict(zip(CAT28, map(float, features[row.signal_row_index])))
        shap_status, contributions = explanation_for_case(row, explanations)
        figure = f"case_{row.case_number:02d}.png"
        fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
        minute = np.arange(150) * 2 / 60
        for axis, channel, label in zip(axes, signal, ("FHR", "TOCO")):
            axis.plot(minute, channel, linewidth=1.2, color="#327A9B")
            axis.set(ylabel=label, xlim=(0, 5))
            axis.grid(alpha=.2)
        axes[-1].set_xlabel("Minutes within stored window")
        fig.suptitle(f"Case {row.case_number:02d} / {row.category} / score={row.score:.4f} / threshold={row.threshold:.4f}")
        fig.tight_layout()
        fig.savefig(out / figure, dpi=130)
        plt.close(fig)
        record = dict(case_number=row.case_number, category=row.category, record_id=row.record_id,
                      mother_id=getattr(row, "mother_id", ""), seg_idx=row.seg_idx,
                      signal_row_index=row.signal_row_index, target=int(row.target), predicted_label=int(row.predicted_label),
                      score=float(row.score), threshold=float(row.threshold), figure=figure,
                      shap_status=shap_status, top_shap=json.dumps(contributions, ensure_ascii=False, allow_nan=False))
        record.update({"value_" + feature: value for feature, value in factor_values.items()})
        rows.append(record)
        markup.extend([f"<section><h2>Case {row.case_number:02d}: {row.category} — {CATEGORY_NAMES[row.category]}</h2>",
            "<p>기록: " + html.escape(str(row.record_id)) + "; 산모: " + html.escape(str(getattr(row, "mother_id", ""))) +
            f"; 구간: {int(row.seg_idx)}; 실제 라벨: {int(row.target)}; 예측 라벨: {int(row.predicted_label)}; " +
            f"점수: {row.score:.6f}; 검증 임계값: {row.threshold:.6f}</p>",
            f"<img src='{figure}' alt='Stored FHR and TOCO traces for case {row.case_number:02d}'>"])
        markup.append("<h3>Cat28 입력 인자 — 28개 전체</h3><p>해당 기록·구간에 정렬된 저장 인자값입니다. SHAP 표본과 독립적이며, 인자값 자체가 예측 기여의 크기를 뜻하지는 않습니다.</p>")
        markup.append(pd.DataFrame({"feature": CAT28, "value": list(factor_values.values())})
                      .to_html(index=False, float_format=lambda value: f"{value:.6g}"))
        if contributions:
            markup.append("<p>기존 SHAP 표본에 포함된 사례: 절댓값 기준 상위 5개 기여</p>")
            markup.append(pd.DataFrame(contributions).to_html(index=False, float_format=lambda value: f"{value:.5g}", na_rep="unavailable"))
        else:
            markup.append("<p>개별 SHAP 기여: 미산출 — 사전 계산 표본에 없거나 유효한 기여값이 없습니다. 파형·오류 사례 선택은 그대로 유지합니다.</p>")
        markup.append("</section>")
    columns = ["case_number", "category", "record_id", "mother_id", "seg_idx", "signal_row_index", "target", "predicted_label",
               "score", "threshold", "figure"] + ["value_" + feature for feature in CAT28] + ["shap_status", "top_shap"]
    pd.DataFrame(rows, columns=columns).to_csv(out / "case_review.csv", index=False)
    markup.append("</html>")
    (out / "case_review.html").write_text("\n".join(markup) + "\n", encoding="utf-8")
    return "case_review.html"
