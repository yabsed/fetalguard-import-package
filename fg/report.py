"""Self-contained HTML/Markdown report and static research figures."""
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .common import read_json, table, write_json
from .supplementary import calibration_bins
from .export_review import collect_tables
from .case_review import run_case_review


def run_report(run, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    case_review_path = run_case_review(run, out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})
    evidence = collect_tables(run, cfg, screened=False)
    for name in ("hypothesis_evidence", "paired_comparisons", "loso", "feature_quality", "feature_response_train",
                 "normalization_matched", "inference_cost", "outcome_associations", "outcome_reading_cells", "outcome_missingness", "outcome_cohort_descriptors", "emr_matched_seeds", "emr_selection"):
        evidence[name].to_csv(out / (name + ".csv"), index=False)
    data = read_json(run / "data/summary.json")
    metrics = read_json(run / "experiment_a/holdout_metrics.json")
    bfile = run / "experiment_b/metrics.json"
    if bfile.exists():
        metrics.update(read_json(bfile))
    rows = []
    for model, result in metrics.items():
        if result.get("status") == "ok":
            row = dict(model=model, n=result["n"], **result["point"])
            for key, ci in result["ci95"].items():
                row[key + "_lo"], row[key + "_hi"] = ci if ci else (np.nan, np.nan)
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    plot_paths = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{suffix}")
        plt.close(fig)
        plot_paths.append(f"figures/{name}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(comparison) * .28)))
    for ax, key in zip(axes, ("auroc", "auprc")):
        ordered = comparison.sort_values(key)
        ax.barh(ordered.model, ordered[key], color="#327A9B")
        # Percentile intervals need not contain the point estimate; draw absolute endpoints.
        ax.hlines(np.arange(len(ordered)), ordered[key + "_lo"], ordered[key + "_hi"], color="black", lw=1)
        ax.set(xlabel=key.upper(), xlim=(0, 1), title="Shared held-out mothers / 95% cluster CI")
    fig.tight_layout()
    save(fig, "model_comparison")
    importance = table(run / "experiment_a/feature_importance.csv").sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(importance.feature, importance.mean_abs_shap, color="#327A9B")
    ax.set(xlabel="Mean |SHAP| (logit scale)", title="Cat28: predictive contribution, not causal explanation")
    fig.tight_layout()
    save(fig, "feature_importance")
    response = evidence["feature_response_train"]
    if not response.empty:
        selected_features = [f for f in ("stv", "fhr_min", "figo_baseline_var", "n_decel") if f in set(response.feature)]
        if selected_features:
            fig, axes = plt.subplots(1, len(selected_features), figsize=(4 * len(selected_features), 3.5), squeeze=False)
            for ax, feature in zip(axes[0], selected_features):
                sub = response.loc[response.feature == feature].sort_values("bin")
                ax.plot(sub.bin, sub.positive_fraction, "o-", color="#327A9B")
                ax.set(title=feature, xlabel="Training quantile bin", ylabel="Abnormal fraction", ylim=(0, 1))
                for _, value in sub.iterrows():
                    ax.annotate(f"n={int(value.n_segments)}", (value.bin, value.positive_fraction), fontsize=6)
            fig.suptitle("Training-only marginal association; not causal and not a test-set tuning guide")
            fig.tight_layout()
            save(fig, "feature_response_training")
    pred = table(run / "experiment_a/test_predictions.csv")
    if (run / "experiment_b/test_predictions.csv").exists():
        pred = pd.concat([pred, table(run / "experiment_b/test_predictions.csv")])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, sub in pred.groupby("model"):
        if name not in {"cat28", "xgb28", "cnn_small_channel_maxabs", "cnn_medium_channel_maxabs"}:
            continue
        fpr, tpr, _ = roc_curve(sub.target, sub.score)
        precision, recall, _ = precision_recall_curve(sub.target, sub.score)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
        bins = calibration_bins(sub.target, sub.score)
        axes[2].plot([r["mean_prediction"] for r in bins], [r["observed"] for r in bins], "o-", label=name)
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[2].set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "curves_calibration")
    sites = pd.read_csv(run / "data/site_inventory.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sites.site, sites.records, label="All records", color="#93BECF")
    ax.bar(sites.site, sites.positive_records, label="Any abnormal segment", color="#D27752")
    ax.set(ylabel="Records", title="Source site composition")
    ax.legend()
    save(fig, "site_composition")
    loso = read_json(run / "supplementary/loso_metrics.json")
    lr = [{"site": site, "auroc": result["raw"]["point"]["auroc"], "brier_raw": result["raw"]["point"]["brier"],
           "brier_platt": result["platt"]["point"]["brier"]} for site, result in loso.items() if result.get("status") == "ok"]
    if lr:
        df = pd.DataFrame(lr)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(df.site, df.auroc, color="#327A9B")
        axes[0].set(title="Leave-one-site-out", ylabel="AUROC", ylim=(0, 1))
        axes[1].plot(df.site, df.brier_raw, "o-", label="Raw")
        axes[1].plot(df.site, df.brier_platt, "o-", label="Source-only Platt")
        axes[1].set(title="Calibration transfer", ylabel="Brier score (lower is better)")
        axes[1].legend()
        save(fig, "loso")
    bstatus = read_json(run / "experiment_b/status.json")
    title = "MOCK TEST — 학술 결론에 사용 금지" if cfg["profile"] == "mock" else "본선 CTG 자동 분석 결과"
    coverage = read_json(run / "supplementary/outcomes.json")
    outcome_rows = []
    for name, result in coverage.items():
        if result.get("status") != "ok":
            outcome_rows.append(dict(outcome=name, status=result.get("status")))
            continue
        delta = result["paired_increment"]["metrics"]["auroc"]
        outcome_rows.append(dict(outcome=name, status="ok", records=result["n"], positive=result["positives"],
            clinical_auroc=result["metrics"]["clinical"]["point"]["auroc"],
            plus_reading_auroc=result["metrics"]["clinical_plus_reading"]["point"]["auroc"],
            delta_auroc=delta["difference"], delta_ci95=str(delta["ci95"])))
    outcome_table = pd.DataFrame(outcome_rows)
    outcome_table.to_csv(out / "outcome_summary.csv", index=False)
    annotation_table = table(run / "supplementary/figo_annotation_agreement.csv")
    emr_result = read_json(run / "supplementary/emr_added.json")
    coverage_rows = [dict(section="A: H1/H2/H3", status="complete"), dict(section="B: H4", status=bstatus["status"])]
    coverage_rows.extend(dict(section="H5: " + k, status=v.get("status")) for k, v in coverage.items())
    coverage_rows.extend(dict(section="LOSO: " + k, status=v.get("status")) for k, v in loso.items())
    official_rows = []
    xgb_file, yolo_file = run / "official/xgboost_emergency_metrics.json", run / "official/yolo_metrics.json"
    if xgb_file.exists():
        for mode, result in read_json(xgb_file).items():
            coverage_rows.append(dict(section="Official XGBoost: " + mode, status=result.get("status")))
            if result.get("status") == "ok":
                official_rows.append(dict(model="XGBoost / " + mode, task="Emergency", n=result["n"],
                    auroc=result["point"]["auroc"], auprc=result["point"]["auprc"], accuracy=result["accuracy"]))
    if yolo_file.exists():
        result = read_json(yolo_file)
        coverage_rows.append(dict(section="Official YOLO", status=result.get("comparison_status", result.get("status"))))
        if "metrics" in result:
            official_rows.append(dict(model="YOLO / image subset", task="Abnormality", n=result["metrics"]["n"],
                auroc=result["metrics"]["point"]["auroc"], auprc=result["metrics"]["point"]["auprc"]))
    if not cfg["official_models"]:
        coverage_rows.append(dict(section="Official models", status="disabled_by_config"))
    coverage_table = pd.DataFrame(coverage_rows)
    coverage_table.to_csv(out / "analysis_coverage.csv", index=False)
    notes = [
        "현장 전용 보고서입니다. 개인별 예측·설명·원자료 경로로 이어지는 링크가 포함됩니다. 이 internal 폴더를 반출 묶음으로 사용하지 마세요.",
        "산모 단위 분할. 모델 선택·임계값은 검증셋에서 결정. 동일 평가 세그먼트에서 비교.",
        "H1–H3: 파형 인자의 예측 기여. 전문의의 인지 과정을 인과적으로 규명한 결과가 아닙니다.",
        "일반 비교의 주 지표 AP(average precision), 보조 지표 AUROC. H2는 단일 인자 선택 기준과 같은 AUROC가 주 지표이며 AP는 보조입니다. 다수 부가 대비는 탐색적입니다.",
        "H4: 차이의 신뢰구간이 0을 포함한다고 동등성이 입증되는 것은 아닙니다. CNN 정규화 대비는 같은 width·seed 비교표를 따로 확인하세요.",
        "H5 및 기관 전이는 탐색적 부가 분석. pH/Apgar는 결측 가능한 기록 단위 아웃컴입니다.",
        "정상 기록 경보율은 완전 정상 라벨 기록 중 관측 가능한 구간에 한정합니다. 경보 횟수는 양성 5분 창 수이며 연속 경보 병합 횟수가 아닙니다.",
        "공식 XGBoost의 타깃은 Emergency입니다. 공식 YOLO는 이미지가 있는 공통 부분집합에서 별도 비교합니다. 배포 가중치의 학습 중복 여부는 확인되지 않았습니다.",
        "FG 인자는 A.2의 30초 평활을 적용합니다. 0.5Hz 자료의 지연 탐색 해상도는 2초이며 5분 창의 장기 감속·기저선 해석에는 제약이 있습니다.",
        "H2의 단일 인자는 선형 및 비선형 스플라인 대조군을 구분합니다. Cat18/Cat28 대비와 평활 민감도 대비는 각각 하나의 설계 요소를 비교합니다.",
        "추론 시간의 측정 범위를 확인하세요. CNN은 원시 창 정규화와 추론을 포함하고 파일 읽기·보간은 제외합니다. 특성 행렬 예측과 인자 추출은 별도 측정하며 합계를 실측 종단 간 지연으로 주장하지 않습니다.",
        "H5의 판독 요약은 전문의 라벨입니다. 모델 예측값을 임상 모형에 추가한 효과로 해석하지 마세요.",
        "CSV 예측·SHAP·체크포인트·원시 신호 캐시는 개인 단위 파생자료입니다. 이 폴더를 그대로 반출하지 말고 기관 심의를 거치세요.",
    ]
    if bstatus.get("hit_cap"):
        notes.append(f"CNN {bstatus['hit_cap']}회가 epoch 상한에 도달했습니다. 수렴 입증으로 해석하지 마세요.")
    if data["bbox_unverified"]:
        notes.append(f"Bbox 또는 이미지 부재로 양성 기록 {data['bbox_unverified']}건의 좌표 동등성은 확인하지 못했습니다.")
    if data["bbox_failed"]:
        notes.append(f"Bbox 검증 실패 {data['bbox_failed']}건: 좌표 동등성은 전량 성립하지 않습니다. 분석 대상은 Abnormality 문자열입니다.")
    brief = comparison[["model", "auroc", "auprc", "sensitivity", "specificity", "ppv", "normal_record_alarm_rate", "false_alarms_per_normal_hour"]]
    artifacts = sorted(p for p in run.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".png", ".pdf"} and ".state" not in p.parts and "yolo_record_cache" not in p.parts)
    links = [(str(p.relative_to(run)), "../" + str(p.relative_to(run))) for p in artifacts]
    markup = ["<!doctype html><meta charset='utf-8'><title>CTG report</title>",
        "<style>body{font:16px system-ui;margin:3em auto;max-width:1200px;color:#20303c}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}.note{background:#f2f6f8;padding:1em}a{color:#176887}</style>",
        f"<h1>현장 전용 — {html.escape(title)}</h1>", f"<p>{data['records']:,} records / {data['mothers']:,} mothers / {data['segments']:,} usable segments</p>",
        "<div class='note'>" + "".join(f"<p>{html.escape(n)}</p>" for n in notes) + "</div>",
        f"<p><a href='{html.escape(case_review_path, quote=True)}'>현장 전용 사례 검토: TP/TN/FP/FN 파형·추출 인자·가용 SHAP</a></p>",
        "<h2>연구 질문과 증거</h2>", evidence["hypothesis_evidence"].to_html(index=False, na_rep="미산출", float_format=lambda x: f"{x:.4f}"),
        "<h2>Held-out comparison</h2>", brief.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Paired differences / 95% mother-cluster CI</h2>", evidence["paired_comparisons"].to_html(index=False, na_rep="미산출", float_format=lambda x: f"{x:.4f}"),
        "<h2>Measurement quality</h2>", evidence["feature_quality"].to_html(index=False),
        "<h2>Matched normalization: identical width and seed</h2>", evidence["normalization_matched"].to_html(index=False),
        "<h2>Inference cost: inspect measurement scope</h2>", evidence["inference_cost"].to_html(index=False),
        "<h2>LOSO operating performance and normal-record alarms</h2>", evidence["loso"].to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Figures</h2>"]
    markup.extend(f"<img src='{html.escape(p)}' alt='{html.escape(Path(p).stem)}'>" for p in plot_paths)
    markup += ["<h2>Outcome coverage</h2>", "<pre>" + html.escape(str({k: v.get('status') for k, v in coverage.items()})) + "</pre>",
               outcome_table.to_html(index=False), "<h2>H5 association (not incremental prediction)</h2>", evidence["outcome_associations"].to_html(index=False),
               evidence["outcome_reading_cells"].to_html(index=False), "<h2>H5 observed vs missing/invalid outcomes</h2>", evidence["outcome_missingness"].to_html(index=False),
               evidence["outcome_cohort_descriptors"].to_html(index=False),
               "<h2>Record-level FIGO agreement</h2>", annotation_table.to_html(index=False),
               "<h2>Added EMR: paired AUROC increment</h2><pre>" + html.escape(str(emr_result["paired_increment"])) + "</pre>",
               "<h2>EMR candidate/selected seeds and validation AP</h2>", evidence["emr_selection"].to_html(index=False),
               "<h2>Official models (different task/subset; reference only)</h2>", pd.DataFrame(official_rows).to_html(index=False),
               "<h2>Analysis coverage</h2>", coverage_table.to_html(index=False), "<h2>Artifact index</h2><ul>"]
    markup.extend(f"<li><a href='{html.escape(href, quote=True)}'>{html.escape(label)}</a></li>" for label, href in links)
    markup.append("</ul>")
    (out / "report.html").write_text("\n".join(markup), encoding="utf-8")
    md = f"# 현장 전용 — {title}\n\n{data['records']:,} records / {data['mothers']:,} mothers / {data['segments']:,} segments\n\n"
    md += "\n\n".join(notes) + "\n\n" + brief.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    md += f"[현장 전용 사례 검토: TP/TN/FP/FN]({case_review_path})\n\n"
    md += "\n\n".join(f"![{Path(p).stem}]({p})" for p in plot_paths)
    md += "\n\n## H5\n\n" + outcome_table.to_markdown(index=False)
    md += "\n\n## FIGO\n\n" + annotation_table.to_markdown(index=False)
    md += "\n\n## Official reference models\n\n" + pd.DataFrame(official_rows).to_markdown(index=False)
    md += "\n\n## Coverage\n\n" + coverage_table.to_markdown(index=False)
    for name in ("hypothesis_evidence", "paired_comparisons", "feature_quality", "normalization_matched", "inference_cost", "loso",
                 "outcome_associations", "outcome_reading_cells", "outcome_missingness", "outcome_cohort_descriptors", "emr_matched_seeds", "emr_selection"):
        frame = evidence[name]
        md += "\n\n## " + name.replace("_", " ") + "\n\n" + (frame.to_markdown(index=False, floatfmt=".4f") if len(frame.columns) else "미산출")
    (out / "report.md").write_text(md + "\n", encoding="utf-8")
    write_json(out / "artifact_index.json", [label for label, href in links])
