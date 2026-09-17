"""Question-led, onsite reading guide from saved results; no fitting or selection.

These figures use unscreened data, native site codes, actual feature ranges
and per-segment SHAP. The onsite_figures/ directory remains onsite-only even
when it is located inside the export_review workspace.
"""
import html
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .common import read_json, sha256, write_json
from .supplementary import calibration_bins


BLUE, ORANGE, INK, GRAY = "#247b91", "#d26742", "#223649", "#758493"
ONSITE_DIRECTORY = re.compile(r"onsite_figures(?:_[A-Za-z0-9_-]+)?\Z")
ONSITE_IMAGE = re.compile(r"\d{2}_[a-z0-9_]+\.(?:png|pdf)\Z")
FEATURES = {
    "fhr_mean": "평균 심박", "fhr_min": "최저 심박", "fhr_max": "최고 심박", "fhr_sd": "심박 표준편차",
    "stv": "표본 간 변동 (STV 대용치)", "brady_frac": "서맥 비율", "tachy_frac": "빈맥 비율",
    "n_decel": "감속 수", "decel_max_depth": "최대 감속 깊이", "decel_time_frac": "감속 시간 비율",
    "n_accel": "가속 수", "toco_mean": "평균 TOCO", "toco_max": "최대 TOCO", "toco_sd": "TOCO 표준편차",
    "n_contractions": "수축 수", "ft_corr0": "동시 FHR–TOCO 상관", "ft_corr_min": "최저 지연 상관",
    "ft_corr_min_lag_s": "최저 상관의 시차", "figo_baseline": "추정 기저선", "figo_baseline_var": "기저선 변이도",
    "n_early_decel": "조기 감속 수", "n_late_decel": "후기 감속 수", "n_variable_decel": "가변 감속 수",
    "n_severe_decel": "중증 감속 수", "n_prolonged_decel": "지연 감속 수",
    "hist_width": "심박 범위", "hist_median": "심박 중앙값", "hist_mode": "심박 최빈값",
}
MODELS = {
    "best_single": ("단일 인자 · 선형", "Single feature / linear"),
    "best_single_nonlinear": ("단일 인자 · 비선형", "Single feature / spline"),
    "logistic28": ("28인자 · 로지스틱", "28 features / logistic"),
    "cat18": ("18인자 · CatBoost", "18 features / CatBoost"),
    "cat28": ("28인자 · CatBoost", "28 features / CatBoost"),
    "xgb28": ("28인자 · XGBoost", "28 features / XGBoost"),
    "robust": ("16인자 · Robust", "16 features / Robust"),
    "cat28_unsmoothed": ("28인자 · 평활 제거", "28 features / no smoothing"),
}
GROUPS = {
    "signal": ("심박 요약", "Signal summaries"), "baseline": ("기저선·분포", "Baseline / distribution"),
    "variability": ("변동성", "Variability"), "range": ("서맥·빈맥", "Brady / tachy"),
    "events": ("감속·가속", "Decelerations / accelerations"),
    "contractions_coupling": ("수축·신호 결합", "Contractions / coupling"),
}
OUTCOMES = {"ph_lt_7_20": "pH < 7.20", "apgar1_lt_7": "1-min Apgar < 7", "apgar5_lt_7": "5-min Apgar < 7"}


def choose_font():
    """Use an installed Korean font; English fallback needs no downloads."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "HCR Dotum"):
        if name in available:
            return name, True
    return "DejaVu Sans", False


def difference(result, metric, reverse=False):
    """Reverse both the estimate and interval endpoints when flipping a contrast."""
    item = result.get("metrics", {}).get(metric, {})
    value, interval = item.get("difference"), item.get("ci95")
    if reverse:
        value = -value if value is not None else None
        interval = [-interval[1], -interval[0]] if interval else None
    return value, interval


def operating_counts(frame):
    """Apply saved validation thresholds, including ties, without retuning."""
    if frame.empty or frame.duplicated(["record_id", "seg_idx"]).any():
        raise ValueError("Expected nonempty unique Cat28 holdout predictions")
    if not frame.target.isin([0, 1]).all() or not frame.score.between(0, 1).all():
        raise ValueError("Invalid holdout labels or probabilities")
    if not np.isfinite(frame.threshold).all():
        raise ValueError("Invalid saved validation thresholds")
    alarm = frame.score.ge(frame.threshold)
    matrix = confusion_matrix(frame.target, alarm.astype(int), labels=[0, 1])
    normal = frame.loc[frame.record_all_normal.eq(1)].assign(alarm=alarm)
    records = normal.groupby("record_id").alarm.max()
    return matrix, int(records.sum()), len(records)


def reading_rates(result):
    """Record-level outcome fractions, with explicit observed denominators."""
    cells = result.get("reading_outcome_cells", [])
    rates = []
    for reading in (0, 1):
        selected = [c for c in cells if c["reading_positive"] == reading]
        total = sum(c["records"] for c in selected)
        positive = sum(c["records"] for c in selected if c["outcome_positive"] == 1)
        rates.append((positive / total if total else np.nan, positive, total))
    return rates


def native_value(value):
    """Readable feature bounds without collapsing nearby heart-rate bins."""
    digits = 1 if abs(value) >= 10 else 2 if abs(value) >= 1 else 3
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def feature_unit(feature):
    if feature in {"brady_frac", "tachy_frac", "decel_time_frac"}:
        return "0–1"
    if feature in {"ft_corr0", "ft_corr_min"}:
        return "r"
    if feature == "ft_corr_min_lag_s":
        return "s"
    if feature.startswith("toco_"):
        return "TOCO"
    if feature.startswith("n_"):
        return "count / 5 min"
    return "bpm"


class Guide:
    def __init__(self, run, out, cfg, korean, link_base=None):
        self.run, self.out, self.cfg, self.korean = run, out, cfg, korean
        self.link_base = Path(link_base) if link_base is not None else out
        self.cards, self.sources = [], {}
        self.metrics = self.load("experiment_a/holdout_metrics.json")
        self.metrics.update(self.load("experiment_b/metrics.json", optional=True))
        self.pairs = self.load("experiment_a/paired_cat28_minus_comparator.json")
        self.summary = self.load("data/summary.json")
        self.splits = self.load("splits/summary.json")
        self.pred = self.load("experiment_a/test_predictions.csv")
        self.cat = self.pred.loc[self.pred.model.eq("cat28")].copy()
        self.load("run_manifest.json")

    def t(self, ko, en):
        return ko if self.korean else en

    def feature(self, name):
        return FEATURES.get(name, name) if self.korean else name.replace("_", " ")

    def model(self, name):
        return self.t(*MODELS[name]) if name in MODELS else name.replace("_", " ")

    def load(self, name, optional=False):
        path = self.run / name
        if optional and not path.exists():
            return pd.DataFrame() if name.endswith(".csv") else {}
        self.sources[name] = sha256(path)
        if name.endswith(".csv"):
            return pd.read_csv(path, dtype={"record_id": str, "mother_id": str, "site": str}, float_precision="round_trip")
        return read_json(path)

    def canvas(self, title, subtitle, rows=1, cols=2, size=(15, 8.5)):
        fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, facecolor="#ffffff")
        fig.text(.035, .96, title, fontsize=22, weight="bold", color=INK, va="top")
        fig.text(.035, .905, subtitle, fontsize=11, color=GRAY, va="top")
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
        return fig, axes

    def save(self, fig, name, title, explanation):
        budget = self.cfg.get("budget", self.cfg.get("profiles", {}).get(self.cfg.get("profile"), {}))
        mode = "MOCK / " if self.cfg.get("profile") == "mock" else ""
        footer = (mode + self.t("현장 이해용", "Onsite guide") +
                  f" | seeds={budget.get('seeds', [])} | CV={budget.get('cv_folds', '?')} | bootstrap={budget.get('bootstrap', '?')}" +
                  self.t(" | CI 표시는 저장된 95% 산모 군집 구간", " | CI markers: saved 95% mother-cluster intervals"))
        if self.cfg.get("profile") == "mock":
            footer += self.t(" | 실행 검사 전용", " | execution test only")
        fig.text(.035, .08, explanation, fontsize=10, color=INK, va="top", linespacing=1.6)
        fig.text(.035, .025, footer, fontsize=8, color=GRAY)
        fig.tight_layout(rect=(.005, .13, .995, .84), pad=2.0, w_pad=3.5, h_pad=4)
        try:
            for extension in ("png", "pdf"):
                fig.savefig(self.out / f"{name}.{extension}", dpi=160, facecolor="white")
        finally:
            plt.close(fig)
        self.cards.append(dict(name=name, title=title, explanation=explanation))

    def forest(self, ax, labels, values, intervals, xlabel, reference=None, percent=False):
        for y, (value, interval) in enumerate(zip(values, intervals)):
            color = BLUE
            if interval and all(v is not None and np.isfinite(v) for v in interval):
                # Absolute endpoints: percentile intervals need not contain the estimate.
                ax.plot(interval, [y, y], color=color, lw=2, solid_capstyle="round")
            if value is not None and np.isfinite(value):
                ax.scatter(value, y, s=55, color=color, zorder=3)
                label = f"{value:.1%}" if percent else (f"{value:+.2g}" if 0 < abs(value) < .001 else f"{value:+.3f}") if reference == 0 else f"{value:.3f}"
            else:
                label = self.t("미산출", "NA")
            ax.text(1.015, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=10)
        ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel=xlabel, ylim=(len(labels) - .5, -.5))
        if reference is not None:
            ax.axvline(reference, color=GRAY, ls="--", lw=1)
        ax.grid(axis="x", alpha=.18)
        if percent:
            ax.xaxis.set_major_formatter(PercentFormatter(1))

    def cohort(self):
        title = self.t("01  어떤 데이터로 학습하고 평가했나?", "01  What was used for training and evaluation?")
        s = self.summary
        fig, axes = self.canvas(title, self.t(
            f"전체 {s['records']:,}건 · {s['mothers']:,}명 · 5분 구간 {s['segments']:,}개",
            f"{s['records']:,} records / {s['mothers']:,} mothers / {s['segments']:,} five-minute segments"))
        names = [self.t("학습", "Train"), self.t("검증 · 선택", "Validation / selection"), self.t("테스트 · 평가", "Test / evaluation")]
        parts = [self.splits[k] for k in ("train", "val", "test")]
        ax = axes[0, 0]
        totals = np.array([p["segments"] for p in parts])
        positive = np.array([p["positives"] for p in parts])
        ax.barh(names, totals - positive, color=BLUE, label=self.t("정상 판독 구간", "Normal-labelled segments"))
        ax.barh(names, positive, left=totals - positive, color=ORANGE, label=self.t("이상 판독 구간", "Abnormal-labelled segments"))
        for i, p in enumerate(parts):
            ax.text(totals[i] + max(totals) * .025, i, f"{totals[i]:,}\n{p['mothers']:,} " + self.t("명", "mothers"), va="center", fontsize=10)
        ax.set(xlim=(0, max(totals) * 1.32), xlabel=self.t("5분 구간 수", "Five-minute segments"))
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=9)
        axes[0, 1].bar(names, positive / totals, color=ORANGE, width=.55)
        for i, value in enumerate(positive / totals):
            axes[0, 1].text(i, value + .012, f"{value:.1%}", ha="center")
        axes[0, 1].set(ylim=(0, 1), ylabel=self.t("이상 판독 구간 비율", "Abnormal segment fraction"))
        axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
        self.save(fig, "01_cohort", title, self.t("같은 산모의 기록은 한 분할에 묶습니다. 막대는 구간 수이며 오른쪽 숫자는 산모 수입니다.",
            "Records from one mother stay in one split. Bars count segments; labels also show mothers."))

    def performance(self):
        names = [n for n in list(MODELS) + sorted(n for n in self.metrics if n.startswith("cnn_"))
                 if self.metrics.get(n, {}).get("status") == "ok"]
        title = self.t("02  어떤 모델이 판독을 더 잘 예측하나?", "02  Which models better predict expert readings?")
        base = self.metrics["cat28"]
        fig, axes = self.canvas(title, self.t(f"같은 테스트 {base['n']:,}구간 · {base['mothers']:,}명 / 점: 추정값, 선: 95% CI",
            f"Shared test: {base['n']:,} segments / {base['mothers']:,} mothers. Dots: estimates; lines: 95% CI"),
            size=(16, max(8.5, 3.5 + .48 * len(names))))
        for ax, metric, label, reference in zip(axes[0], ["auprc", "auroc"],
                [self.t("AP · 이상 구간을 잘 모으는가 (높을수록 좋음)", "AP / average precision (higher is better)"),
                 self.t("AUROC · 정상과 이상을 잘 구분하는가", "AUROC / discrimination (higher is better)")],
                [base["point"]["prevalence"], .5]):
            self.forest(ax, [self.model(n) for n in names], [self.metrics[n]["point"][metric] for n in names],
                        [self.metrics[n].get("ci95", {}).get(metric) for n in names], label, reference)
            ax.set_xlim(0, 1)
        self.save(fig, "02_model_performance", title, self.t(
            "점선: AP는 테스트 양성 비율, AUROC는 0.5 기준입니다. 모델 선택은 검증셋에서 수행했습니다.\nCNN이 미수행이면 이 그림에 CNN 점수를 표시하지 않습니다.",
            "Dashed lines: test prevalence for AP, 0.5 for AUROC. Models were selected on validation data.\nCNN scores appear only when evaluated."))

    def contrasts(self):
        comparisons = [
            (self.t("조합 − 비선형 단일 인자", "Cat28 - nonlinear single"), self.pairs.get("best_single_nonlinear", {}), False),
            (self.t("28인자 − 18인자", "Cat28 - Cat18"), self.pairs.get("cat18", {}), False),
            (self.t("평활 제거 − 30초 평활", "No smoothing - 30s smoothing"), self.pairs.get("cat28_unsmoothed", {}), True),
        ]
        emr = self.load("supplementary/emr_added.json", optional=True)
        comparisons.append((self.t("EMR 추가 − 파형만", "Added EMR - waveform only"), emr.get("paired_increment", {}), False))
        title = self.t("03  무엇을 추가하거나 바꾸면 좋아지나?", "03  What improves when a component changes?")
        fig, axes = self.canvas(title, self.t("같은 테스트 대상의 짝지은 차이 · 각 행의 뺄셈 방향을 읽으세요", "Paired differences on the same test cohort; read each contrast direction"))
        for ax, metric in zip(axes[0], ("auroc", "auprc")):
            pairs = [difference(r, metric, rev) for _, r, rev in comparisons]
            self.forest(ax, [label for label, _, _ in comparisons], [v for v, _ in pairs], [ci for _, ci in pairs],
                        self.t("성능 차이 · 오른쪽이면 개선", "Performance change / right favors addition") + " (" + ("AP" if metric == "auprc" else "AUROC") + ")", 0)
        self.save(fig, "03_paired_changes", title, self.t("선이 0을 가로지르면 이번 자료에서 차이가 불확실합니다. 동등성을 입증한 것은 아닙니다.\n서로 다른 모델의 CI 겹침 대신, 같은 대상에서 계산한 차이의 CI를 봅니다.",
            "A line crossing zero indicates an inconclusive difference, not equivalence.\nThese are paired difference intervals, not overlap of separate model intervals."))

    def operating(self):
        matrix, alarmed, total = operating_counts(self.cat)
        title = self.t("04  무엇을 놓치고, 얼마나 경보하나?", "04  What is missed, and how often are alarms raised?")
        threshold = self.metrics["cat28"]["threshold"]
        fig, axes = self.canvas(title, self.t(f"Cat28 · 검증셋에서 정한 임계값 {threshold:.4f}를 테스트에 그대로 적용",
            f"Cat28 / saved validation threshold {threshold:.4f}, applied unchanged to test"))
        ax = axes[0, 0]
        denominator = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, denominator, out=np.zeros((2, 2), float), where=denominator > 0)
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        names = [[self.t("정상 유지 (TN)", "Correct normal (TN)"), self.t("오경보 (FP)", "False alarm (FP)")],
                 [self.t("놓친 이상 (FN)", "Missed abnormal (FN)"), self.t("포착한 이상 (TP)", "Detected abnormal (TP)")]]
        for row in range(2):
            for col in range(2):
                percentage = f"{fractions[row, col]:.1%}" if denominator[row, 0] else "NA"
                ax.text(col, row, f"{names[row][col]}\n{matrix[row, col]:,}" + self.t(" 구간", " segments") + f"\n{percentage}",
                        ha="center", va="center", fontsize=15, color="white" if fractions[row, col] > .55 else INK)
        ax.set(xticks=[0, 1], xticklabels=[self.t("모델: 정상", "Model: normal"), self.t("모델: 경보", "Model: alarm")],
               yticks=[0, 1], yticklabels=[self.t("실제 정상", "Actual normal"), self.t("실제 이상", "Actual abnormal")])
        ax = axes[0, 1]
        points = self.metrics["cat28"]["point"]
        keys = ["sensitivity", "ppv", "normal_record_alarm_rate"]
        labels = [self.t("실제 이상 구간 중 포착", "Detected / abnormal segments"),
                  self.t("경보 구간 중 실제 이상", "Abnormal / alarmed segments"),
                  self.t("완전 정상 기록 중 경보 발생", "Alarmed / fully normal records")]
        self.forest(ax, labels, [points[k] for k in keys], [self.metrics["cat28"]["ci95"].get(k) for k in keys],
                    self.t("각각의 분모 중 비율", "Fraction within each denominator"), percent=True)
        ax.set_xlim(0, 1)
        ax.set_title(self.t(f"완전 정상 {total:,}건 중 {alarmed:,}건에 경보", f"Alarms in {alarmed:,} of {total:,} fully normal records"), fontsize=12)
        self.save(fig, "04_errors_and_alarms", title, self.t("왼쪽 %는 실제 라벨별 행 비율입니다. 오른쪽 세 지표는 분모가 서로 다릅니다.\n정상 기록 경보는 그 기록의 관측 가능한 구간에서 한 번 이상 경보한 경우입니다.",
            "Matrix percentages are row-normalized by actual label. The three rates have different denominators.\nA normal record is alarmed if any observed segment triggers an alarm."))

    def scores(self):
        title = self.t("05  점수는 어떻게 겹치고, 확률은 얼마나 맞나?", "05  How do scores overlap, and are probabilities reliable?")
        fig, axes = self.canvas(title, self.t("Cat28 · 테스트 구간 / 점수 분포와 확률 보정을 함께 읽기", "Cat28 test segments / score distribution and calibration"))
        ax = axes[0, 0]
        for target, color, label in [(0, BLUE, self.t("실제 정상", "Actual normal")), (1, ORANGE, self.t("실제 이상", "Actual abnormal"))]:
            values = self.cat.loc[self.cat.target.eq(target), "score"]
            if len(values):
                ax.hist(values, bins=np.linspace(0, 1, 21), weights=np.ones(len(values)) / len(values),
                        histtype="stepfilled", alpha=.4, color=color, label=label + f" (n={len(values):,})")
        ax.axvline(self.metrics["cat28"]["threshold"], ls="--", color=INK, label=self.t("검증 임계값", "Validation threshold"))
        ax.set(xlabel=self.t("모델 점수", "Model score"), ylabel=self.t("각 실제 클래스 안의 구간 비율", "Fraction within each actual class"), xlim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=9)
        ax = axes[0, 1]
        bins = calibration_bins(self.cat.target, self.cat.score)
        ax.plot([0, 1], [0, 1], "--", color=GRAY, label=self.t("예측 확률 = 관측 비율", "Predicted = observed"))
        ax.plot([b["mean_prediction"] for b in bins], [b["observed"] for b in bins], "o-", color=BLUE)
        for b in bins:
            ax.annotate(f"n={b['n']}", (b["mean_prediction"], b["observed"]), xytext=(4, 6), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=self.t("구간별 평균 예측 확률", "Mean predicted probability per bin"),
               ylabel=self.t("실제로 이상 판독된 비율", "Observed abnormal fraction"))
        ax.legend(fontsize=9)
        self.save(fig, "05_scores_and_calibration", title, self.t("점수 분포는 정상·이상 각각 합계 100%입니다. 확률 보정 그림의 점선에 가까울수록 확률과 빈도가 맞습니다.\n보정 곡선은 기술통계이며 점에 CI를 붙이지 않았습니다. 작은 bin의 변동에 주의하세요.",
            "Each class histogram sums to 100%. Calibration close to the diagonal means probabilities match observed frequencies.\nCalibration points are descriptive, without confidence intervals; inspect bin sizes."))

    def curves(self):
        title = self.t("06  탐지율을 높일 때 어떤 대가가 따르나?", "06  What changes as detection increases?")
        fig, axes = self.canvas(title, self.t("같은 테스트에서 Cat28 · Cat18 · 비선형 단일 인자 비교", "Cat28, Cat18 and nonlinear single-feature models on the same test"))
        for model, color in zip(("cat28", "cat18", "best_single_nonlinear"), (BLUE, ORANGE, "#8163a0")):
            sub = self.pred.loc[self.pred.model.eq(model)]
            if sub.empty or sub.target.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub.target, sub.score)
            precision, recall, _ = precision_recall_curve(sub.target, sub.score)
            axes[0, 0].plot(fpr, tpr, label=self.model(model), color=color)
            axes[0, 1].step(recall, precision, where="post", label=self.model(model), color=color)
        axes[0, 0].plot([0, 1], [0, 1], "--", color=GRAY)
        axes[0, 1].axhline(self.cat.target.mean(), ls="--", color=GRAY)
        p = self.metrics["cat28"]["point"]
        axes[0, 0].scatter(1 - p["specificity"], p["sensitivity"], color=INK, s=90, marker="*", zorder=5)
        axes[0, 1].scatter(p["sensitivity"], p["ppv"], color=INK, s=90, marker="*", zorder=5)
        axes[0, 0].set(xlabel=self.t("정상 구간 중 오경보 비율", "False-positive fraction"), ylabel=self.t("이상 구간 중 포착 비율", "Sensitivity"), title="ROC")
        axes[0, 1].set(xlabel=self.t("이상 구간 중 포착 비율", "Recall / sensitivity"), ylabel=self.t("경보 구간 중 실제 이상 비율", "Precision / PPV"), title="Precision–recall")
        for ax in axes[0]:
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=9)
        self.save(fig, "06_detection_tradeoffs", title, self.t("별표는 저장된 검증 임계값에서 Cat28의 운영점입니다. 곡선을 보고 테스트 임계값을 다시 선택하지 않습니다.",
            "Stars show Cat28 at the saved validation threshold. These curves do not select a new test threshold."))

    def contributions(self):
        importance = self.load("experiment_a/feature_importance.csv").sort_values("mean_abs_shap", ascending=False)
        top = importance.head(12).iloc[::-1]
        title = self.t("07  중요한 인자와 꼭 필요한 인자군은 같은가?", "07  Contribution and group removal answer different questions")
        fig, axes = self.canvas(title, self.t("왼쪽: Cat28의 설명상 기여 / 오른쪽: 인자군을 제거하고 다시 학습한 성능 차이",
            "Left: Cat28 explanation magnitude. Right: performance lost after removing and refitting a feature group"), size=(17, 10))
        axes[0, 0].barh([self.feature(f) for f in top.feature], top.mean_abs_shap, color=BLUE)
        axes[0, 0].set(xlabel=self.t("평균 |SHAP| · logit 단위", "Mean |SHAP| / logit units"), title=self.t("기여 상위 12개", "Top 12 contributors"))
        groups = list(GROUPS)
        pairs = [difference(self.pairs.get("without_" + g, {}), "auprc") for g in groups]
        self.forest(axes[0, 1], [self.t(*GROUPS[g]) for g in groups], [v for v, _ in pairs], [ci for _, ci in pairs],
                    self.t("Cat28 AP − 해당 군 제거 AP", "Cat28 AP - AP after removal"), 0)
        self.save(fig, "07_contribution_and_ablation", title, self.t("SHAP가 커도 다른 인자가 그 정보를 대신할 수 있습니다. 오른쪽 양수는 그 군을 제거했을 때의 성능 감소입니다.\nSHAP는 모델 예측에 대한 기여이며, 원인 효과나 전문의 사고 과정의 증거가 아닙니다.",
            "A large SHAP contribution can be redundant. Positive removal differences indicate lost AP.\nSHAP explains model predictions; it does not establish causal effects or clinician reasoning."))
        return importance

    def shap_directions(self, importance):
        frame = self.load("experiment_a/local_explanations.csv", optional=True)
        selected = [f for f in importance.head(12).feature if "shap_" + f in frame and "value_" + f in frame]
        if not selected:
            return
        title = self.t("08  인자 값이 높을 때 예측은 어느 쪽으로 움직이나?", "08  Which direction does each feature push a prediction?")
        fig, axes = self.canvas(title, self.t(f"저장된 SHAP 표본 {len(frame):,}구간 · 점 하나는 한 구간 · 색은 인자 내 값의 상대 순위",
            f"{len(frame):,} saved SHAP segments. One dot per segment; color is within-feature value rank"), cols=1, size=(15, 10))
        ax, rng = axes[0, 0], np.random.default_rng(42)
        for y, feature in enumerate(selected):
            value = frame["value_" + feature].rank(pct=True)
            ax.scatter(frame["shap_" + feature], y + rng.uniform(-.25, .25, len(frame)), c=value,
                       cmap="coolwarm", vmin=0, vmax=1, s=8, alpha=.65, linewidths=0, rasterized=True)
        ax.axvline(0, color=GRAY, lw=1)
        ax.set(yticks=range(len(selected)), yticklabels=[self.feature(f) for f in selected],
               ylim=(len(selected) - .5, -.5), xlabel=self.t("정상 쪽  ←  SHAP (logit)  →  이상 쪽", "Toward normal  <-  SHAP (logit)  ->  toward abnormal"))
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="coolwarm")
        bar = fig.colorbar(sm, ax=ax, fraction=.025, pad=.03)
        bar.set_ticks([0, 1], labels=[self.t("값 낮음", "Low value"), self.t("값 높음", "High value")])
        self.save(fig, "08_shap_direction", title, self.t("점의 세로 흩어짐은 겹침을 줄이기 위한 고정 난수입니다. 색은 인자별 순위여서 서로 다른 인자의 크기를 비교하지 않습니다.\n같은 색이 양쪽에 나타날 수 있습니다. 다른 인자와의 조합에 따라 예측 기여가 달라지기 때문입니다.",
            "Vertical jitter is deterministic and only separates overlapping dots. Colors compare ranks within each feature.\nA value can push predictions in different directions depending on the other features."))

    def responses(self, importance):
        response = self.load("experiment_a/feature_response_train.csv")
        selected = [f for f in importance.feature if f in set(response.feature)][:8]
        for page, start in enumerate(range(0, len(selected), 4), 1):
            title = self.t(f"09.{page}  인자 값에 따라 실제 판독 비율은 어떻게 변하나?",
                           f"09.{page}  How does the reading fraction vary with feature values?")
            fig, axes = self.canvas(title, self.t("훈련 데이터만 · 실제 값 범위별 이상 판독 비율 · SHAP 상위 인자",
                "Training data only / abnormal fractions by actual feature ranges / leading SHAP features"), 2, 2, (17, 12))
            for ax, feature in zip(axes.flat, selected[start:start + 4]):
                sub = response.loc[response.feature.eq(feature)].sort_values("bin")
                x = np.arange(len(sub))
                ax.plot(x, sub.positive_fraction, "o-", color=BLUE)
                ax.axhline(self.splits["train"]["positives"] / self.splits["train"]["segments"], ls="--", color=GRAY)
                for i, row in enumerate(sub.itertuples()):
                    ax.annotate(f"n={row.n_segments:,}", (i, row.positive_fraction), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
                ax.set(xticks=x, xticklabels=[f"{native_value(r.lower)}–\n{native_value(r.upper)}" for r in sub.itertuples()],
                       ylim=(0, 1), title=self.feature(feature), ylabel=self.t("이상 판독 비율", "Abnormal fraction"),
                       xlabel=feature + " (" + feature_unit(feature) + ")" + self.t(" · 값 구간", " / value bins"))
                ax.tick_params(axis="x", labelsize=8)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            for ax in list(axes.flat)[len(selected[start:start + 4]):]:
                ax.set_visible(False)
            self.save(fig, f"09_feature_response_{page}", title, self.t("가로축은 동일 간격으로 배치한 값 구간이며 실제 값 사이 거리를 뜻하지 않습니다. 점선은 훈련 양성 비율입니다.\n주변 연관성으로 다른 인자를 보정하지 않았습니다. STV는 표본 간 차이 대용치이며 임상 STV와 다릅니다.",
                "Bins are equally spaced categories, not equally spaced values. Dashed line: training prevalence.\nThese marginal associations are unadjusted. STV here is a sampled-difference proxy, not clinical STV."))

    def quality(self):
        frame = self.load("data/feature_quality.csv").sort_values("zero_fraction", ascending=False, kind="stable")
        title = self.t("10  계산된 0은 정말 사건이 없다는 뜻인가?", "10  Does a computed zero mean the event was absent?")
        fig, axes = self.canvas(title, self.t("전체 사용 구간의 인자별 0 비율 · 값이 항상 같거나 창 길이로 측정 불가한 항목 표시",
            "Zero fractions across usable segments; constant and structurally unobservable features flagged"), cols=1, size=(16, 14))
        ax = axes[0, 0]
        colors = [ORANGE if s != "ok" else BLUE for s in frame.status]
        ax.barh(range(len(frame)), frame.zero_fraction, color=colors)
        for y, row in enumerate(frame.itertuples()):
            status = self.t(" / 측정 불가", " / unobservable") if row.status == "structurally_unobservable" else self.t(" / 상수", " / constant") if row.status == "constant" else ""
            ax.text(1.015, y, f"{row.zero_fraction:.1%}{status}", va="center", fontsize=9)
        ax.set(yticks=range(len(frame)), yticklabels=[self.feature(f) for f in frame.feature], xlim=(0, 1.32),
               ylim=(len(frame) - .5, -.5), xlabel=self.t("값이 0인 구간의 비율", "Fraction of segments equal to zero"))
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        self.save(fig, "10_measurement_quality", title, self.t("0은 실제 부재, 추출 규칙, 관측 창의 한계에서 나올 수 있습니다. '상수'와 '측정 불가'는 구분해야 합니다.\n인자별 단위가 달라 평균 크기를 하나의 순위로 비교하지 않습니다.",
            "Zero can reflect absence, extraction rules or window limits. Constant and unobservable are different conditions.\nFeatures have different units; their raw means are not ranked against each other."))

    def sites(self):
        raw = self.load("supplementary/loso_metrics.json", optional=True)
        sites = sorted(s for s, r in raw.items() if r.get("status") == "ok")
        if not sites:
            return
        title = self.t("11  사이트가 달라지면 경보 부담도 달라지나?", "11  Does the alarm burden change across sites?")
        fig, axes = self.canvas(title, self.t("각 사이트를 학습에서 제외한 LOSO · 원 사이트 코드 · 기준 임계값은 source 검증셋에서 선택",
            "Leave-one-site-out / native site codes / thresholds from source validation only"), cols=3, size=(19, 10))
        labels = [f"{s}  (M={raw[s]['raw']['mothers']:,})" for s in sites]
        for ax, metric, label in zip(axes[0], ["prevalence", "specificity", "normal_record_alarm_rate"],
                [self.t("이상 판독 비율 · 구간", "Abnormal fraction / segments"), self.t("특이도 · 정상 구간", "Specificity / normal segments"),
                 self.t("경보율 · 완전 정상 기록", "Alarm fraction / fully normal records")]):
            self.forest(ax, labels, [raw[s]["raw"]["point"][metric] for s in sites],
                        [raw[s]["raw"]["ci95"].get(metric) for s in sites], label, .9 if metric == "specificity" else None, True)
            ax.set_xlim(0, 1)
        self.save(fig, "11_site_alarm_burden", title, self.t("M은 해당 사이트의 평가 산모 수입니다. 지표별 분모는 제목처럼 다르며 특이도 점선은 목표 90%입니다.\n이 사이트 코드는 실제 병원과 일대일로 확인된 식별자가 아닙니다. 적은 표본의 넓은 CI도 함께 읽으세요.",
            "M counts evaluated mothers. Denominators differ by metric. The specificity target is 90%.\nSite codes are not verified hospital identities. Read the uncertainty for small sites."))
        title = self.t("12  확률 보정은 무엇을 바꾸나?", "12  What does probability calibration change?")
        fig, axes = self.canvas(title, self.t("LOSO · 같은 대상에서 보정 전후 비교 · source-only Platt", "LOSO / same-cohort raw versus source-only Platt calibration"), size=(17, 10))
        for ax, metric, label in zip(axes[0], ["brier", "normal_record_alarm_rate"],
                [self.t("Brier · 확률 오차 (낮을수록 좋음)", "Brier / probability error (lower is better)"),
                 self.t("완전 정상 기록 경보율", "Normal-record alarm fraction")]):
            for y, site in enumerate(sites):
                before, after = [raw[site][arm]["point"].get(metric) for arm in ("raw", "platt")]
                if before is None or after is None:
                    ax.text(.5, y, "NA", va="center")
                    continue
                ax.plot([before, after], [y, y], color=GRAY, lw=2)
                ax.scatter(before, y, marker="o", facecolors="white", edgecolors=BLUE, s=100, label=self.t("보정 전", "Raw") if y == 0 else None, zorder=3)
                ax.scatter(after, y, marker="x", color=ORANGE, s=65, label=self.t("보정 후", "Platt") if y == 0 else None, zorder=4)
            ax.set(yticks=range(len(sites)), yticklabels=sites, ylim=(len(sites) - .5, -.5), xlabel=label, xlim=(0, 1))
            ax.legend(fontsize=10)
        self.save(fig, "12_site_calibration", title, self.t("빈 원과 ×가 겹치면 값이 같습니다. 왼쪽은 확률 오차, 오른쪽은 경보 부담으로 서로 다른 질문입니다.\n그림의 선은 보정 전후 점추정값 연결이며 CI가 아닙니다. 보정은 제외 사이트의 라벨을 사용하지 않습니다.",
            "Overlapping circle and cross mean equal values. Probability error and alarm burden are distinct outcomes.\nConnecting lines are not confidence intervals. Calibration never uses held-site labels."))

    def outcomes(self):
        results = self.load("supplementary/outcomes.json", optional=True)
        names = [n for n in OUTCOMES if n in results]
        if not names:
            return
        title = self.t("13  이상 판독과 아웃컴은 어떻게 연결되나?", "13  How do expert readings relate to outcomes?")
        fig, axes = self.canvas(title, self.t("기록 단위 · 실제 전문의 판독 요약 · 아웃컴이 관측된 기록에서 비교", "Record-level expert readings / compare records with observed outcomes"), cols=len(names), size=(18, 9))
        finite_rates = [v for n in names for v, _, _ in reading_rates(results[n]) if np.isfinite(v)]
        upper = min(1.15, max(.15, max(finite_rates, default=0) * 1.5))
        for ax, name in zip(axes[0], names):
            result = results[name]
            rates = reading_rates(result)
            for x, (rate, positive, n) in enumerate(rates):
                if n:
                    ax.bar(x, rate, color=[BLUE, ORANGE][x], width=.55)
                    ax.text(x, rate + upper * .025, f"{rate:.1%}\n{positive:,}/{n:,}", ha="center", fontsize=12)
                else:
                    ax.text(x, .08, self.t("미산출", "NA"), ha="center")
            cohorts = result.get("cohorts", {})
            observed = cohorts.get("observed_valid", {}).get("records", 0)
            missing = cohorts.get("missing_or_invalid", {}).get("records", 0)
            ax.set(title=OUTCOMES[name], xticks=[0, 1], xticklabels=[self.t("이상 판독 없음", "No abnormal reading"), self.t("이상 판독 있음", "Any abnormal reading")],
                   ylim=(0, upper), xlim=(-.6, 1.6), ylabel=self.t("아웃컴 사건 비율", "Outcome event fraction"),
                   xlabel=self.t(f"관측 {observed:,}건 / 결측·범위 밖 {missing:,}건", f"Observed {observed:,} / missing-invalid {missing:,}"))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        self.save(fig, "13_readings_and_outcomes", title, self.t("막대 위에는 사건 수/분모를 표시했습니다. 관측된 기록의 비보정 연관성이며 인과 효과를 뜻하지 않습니다.\n이상 판독 요약은 전문의 라벨입니다. 모델이 예측한 위험 점수를 사용한 분석이 아닙니다.",
            "Labels show events/records. These are unadjusted associations among observed records, not causal effects.\nReading summaries come from expert labels, not model-predicted risk."))
        title = self.t("14  판독 요약을 추가하면 아웃컴 예측도 좋아지나?", "14  Does adding reading summaries improve outcome prediction?")
        fig, axes = self.canvas(title, self.t("임상 정보 + 판독 요약 − 임상 정보만 · 산모 그룹 OOF의 짝지은 차이", "Clinical plus reading minus clinical only / paired mother-grouped OOF differences"), cols=3, size=(18, 8.5))
        for ax, metric in zip(axes[0], ("auroc", "auprc", "brier")):
            pairs = [difference(results[n].get("paired_increment", {}), metric) for n in names]
            label = ("AP" if metric == "auprc" else metric.upper()) + self.t(" 차이", " difference")
            label += self.t(" · 왼쪽이면 개선", " / left is better") if metric == "brier" else self.t(" · 오른쪽이면 개선", " / right is better")
            self.forest(ax, [OUTCOMES[n] for n in names], [v for v, _ in pairs], [ci for _, ci in pairs], label, 0)
        self.save(fig, "14_outcome_increment", title, self.t("연관성과 예측 증분은 다릅니다. Brier는 낮을수록 좋으므로 개선 방향이 반대입니다.\n아웃컴이 관측된 기록의 OOF 평가로, 앞선 5분 구간 holdout과 평가 대상·단위가 다릅니다.",
            "Association differs from incremental prediction. Lower Brier is better, so its improvement direction is reversed.\nThis record-level observed-outcome OOF cohort differs from the segment holdout above."))

    def index(self):
        cfg = self.cfg
        budget = cfg.get("budget", cfg.get("profiles", {}).get(cfg.get("profile"), {}))
        title = "현장 결과 읽기"  # HTML uses browser font fallback even without a plotting font.
        intro = "데이터 → 모델 비교 → 경보와 오류 → 인자 해석 → 사이트 → 아웃컴 순서로 읽습니다. 각 그림 아래에 읽는 법을 붙였습니다."
        status = "CNN 수행" if cfg.get("cnn") else "CNN 미수행 · H4 미검증"
        status += " / 공식 모델 " + ("수행 설정" if cfg.get("official_models") else "미수행")
        context = f"{cfg.get('run_label', cfg.get('profile', ''))} · seeds={budget.get('seeds', [])} · CV={budget.get('cv_folds')} · bootstrap={budget.get('bootstrap')}"
        markup = ["<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
                  f"<title>{title}</title>", "<style>body{font:16px/1.65 system-ui,sans-serif;color:#223649;background:#f5f7fa;margin:0}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:36px}nav{columns:2}section{background:white;padding:24px;margin:30px 0;border-radius:12px}img{width:100%;height:auto}a{color:#176e87}.meta{color:#526475}figcaption{white-space:pre-line}small{font-size:13px}@media(max-width:700px){nav{columns:1}main{padding:12px}section{padding:12px}}</style><main>",
                  f"<h1>{title}</h1><p>{intro}</p>", f"<p class='meta'>{html.escape(context)}<br>{status}</p>",
                  "<p><small>현장 전용: 개별 구간 SHAP·실제 값 범위·원 사이트 코드를 포함합니다.</small></p>"]
        if cfg.get("profile") == "mock":
            markup.append("<p><strong>MOCK — 실행 검사 결과입니다. 성능을 연구 결론으로 사용하지 마세요.</strong></p>")
        if cfg.get("analysis_role"):
            markup.append(f"<p class='meta'>{html.escape(cfg['analysis_role'])}</p>")
        relative = Path(os.path.relpath(self.run.parent / "export_review/visit_audit/index.html", self.link_base)).as_posix()
        markup.append(f"<p><a href='{html.escape(quote(relative), quote=True)}'>반출 검토용 데이터 지도 · 학습 종료 진단 · 다음 방문 준비표</a></p>")
        for path, label in [("report.html", "전체 수치·분석 보고서"), ("case_review.html", "TP/TN/FP/FN 실제 파형 사례")]:
            if (self.run / "report" / path).exists():
                relative = Path(os.path.relpath(self.run / "report" / path, self.link_base)).as_posix()
                markup.append(f"<a href='{html.escape(quote(relative), quote=True)}'>{label}</a> · ")
        markup.append("<nav><ol>")
        markup += [f"<li><a href='#{c['name']}'>{html.escape(c['title'])}</a></li>" for c in self.cards]
        markup.append("</ol></nav>")
        for card in self.cards:
            name, title = card["name"], html.escape(card["title"])
            markup += [f"<section id='{name}'><h2>{title}</h2><figure><a href='{name}.png'><img loading='lazy' src='{name}.png' alt='{title}'></a>",
                       f"<figcaption>{html.escape(card['explanation'])}</figcaption></figure><a href='{name}.png'>PNG</a> · <a href='{name}.pdf'>PDF</a> · <a href='#'>목차로</a></section>"]
        markup.append("</main></html>")
        (self.out / "index.html").write_text("\n".join(markup), encoding="utf-8")
        (self.out / "README.md").write_text("# 현장 결과 읽기\n\n[index.html](index.html)을 먼저 여세요.\n\n" + intro + "\n\n" + context + "\n\n" + status +
            "\n\n저장된 결과로만 생성합니다. 각 PNG/PDF는 같은 그림입니다. SHAP 점·실제 범위·사이트 코드가 있는 내부용 자료입니다.\n", encoding="utf-8")
        # Detect concurrent modification rather than attributing mixed results to one source.
        for name, digest in self.sources.items():
            if sha256(self.run / name) != digest:
                raise ValueError(f"Source changed during figure generation: {name}")
        write_json(self.out / "manifest.json", dict(format=1, purpose="onsite_understanding",
            review_status="onsite_only_not_screened", language="ko" if self.korean else "en",
            source_files=self.sources, generator_sha256=sha256(__file__), figures=self.cards,
            files={p.name: sha256(p) for p in sorted(self.out.iterdir()) if p.is_file() and p.name != "manifest.json"}))


def run_onsite_figures(run, out, cfg, *, link_base=None):
    run, out = Path(run).resolve(), Path(out).resolve()
    reserved = {"data", "splits", "experiment_a", "experiment_b", "supplementary", "official", "report"}
    if (run.name != "internal" or out.parent != run or not (run / "run_manifest.json").is_file()
            or out.name.startswith(".") or out.name in reserved):
        raise ValueError("Onsite figures must be a dedicated directory directly inside internal/")
    out.mkdir(parents=True, exist_ok=True)
    font, korean = choose_font()
    with plt.rc_context({"font.family": [font, "DejaVu Sans"], "font.size": 11, "axes.unicode_minus": False,
                         "axes.labelcolor": INK, "text.color": INK, "axes.titlepad": 16, "savefig.bbox": None}):
        guide = Guide(run, out, cfg, korean, link_base=link_base)
        guide.cohort()
        guide.performance()
        guide.contrasts()
        guide.operating()
        guide.scores()
        guide.curves()
        importance = guide.contributions()
        guide.shap_directions(importance)
        guide.responses(importance)
        guide.quality()
        guide.sites()
        guide.outcomes()
        guide.index()
    return out / "index.html"


def validate_onsite_figures(out, run=None):
    """Validate the separate onsite manifest; this never grants export screening."""
    out = Path(out)
    paths = list(out.rglob("*"))
    if out.is_symlink() or any(p.is_symlink() for p in (*out.parents, *paths)):
        raise ValueError("Symbolic links are not allowed in onsite figures")
    actual = {p.name for p in paths if p.is_file()}
    if any(not p.is_file() or p.parent != out for p in paths):
        raise ValueError("Unexpected nested onsite figure directory")
    if not {"manifest.json", "index.html", "README.md"}.issubset(actual):
        raise ValueError("Incomplete onsite figure manifest")
    if any(n not in {"manifest.json", "index.html", "README.md"} and not ONSITE_IMAGE.fullmatch(n) for n in actual):
        raise ValueError("Unexpected file in onsite figures")
    guide = read_json(out / "manifest.json")
    if guide.get("purpose") != "onsite_understanding":
        raise ValueError("Invalid onsite purpose")
    # Older guides did not record review_status, but are still strictly onsite-only.
    if guide.get("review_status", "onsite_only_not_screened") != "onsite_only_not_screened":
        raise ValueError("Onsite figures cannot claim screening approval")
    if set(guide["files"]) != actual - {"manifest.json"}:
        raise ValueError("Onsite manifest inventory mismatch")
    for name, digest in guide["files"].items():
        if sha256(out / name) != digest:
            raise ValueError("Onsite figure checksum mismatch")
    for name, digest in guide["source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Invalid onsite source path")
        if run is not None:
            path = Path(run) / relative
            if path.is_symlink() or any(p.is_symlink() for p in path.parents) or sha256(path) != digest:
                raise ValueError("Onsite source checksum mismatch")
    return {"status": "onsite_only_not_screened", "files": len(actual)}


def build_onsite_figures(run, out, cfg):
    """Atomically publish a guide with links relative to its final destination."""
    run, out = Path(run).absolute(), Path(out).absolute()
    if (run.name != "internal" or not ONSITE_DIRECTORY.fullmatch(out.name)
            or out.parent not in (run, run.parent / "export_review")):
        raise ValueError("Choose an onsite_figures directory inside internal/ or export_review/")
    if out.is_symlink() or any(p.is_symlink() for p in (*out.parents, run, *run.parents)):
        raise ValueError("Symbolic-link onsite destinations are not accepted")
    if out.exists() and any(out.iterdir()):
        # Recover a crash after publication but before the runner wrote its marker.
        validate_onsite_figures(out, run)
        return out / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onsite-build-", dir=run) as temporary:
        staging = Path(temporary)
        run_onsite_figures(run, staging, cfg, link_base=out)
        validate_onsite_figures(staging, run)
        if out.exists():
            out.rmdir()  # The runner may have pre-created this empty target.
        staging.rename(out)
    return out / "index.html"
