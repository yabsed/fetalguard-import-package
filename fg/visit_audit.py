"""First-visit decision support from private saved evidence; never fits models.

Training diagnostics read training/validation histories only, never test scores.
HTML/SVG uses the standard library so failed ML preflight still has a report.
"""
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from .survey import write_csv
from .telemetry import save_json, stamp


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def budget_diagnosis(count, best, cap, patience, complete=True):
    count, best, cap, patience = map(finite, (count, best, cap, patience))
    stale = count - best if count is not None and best is not None else None
    hit_cap = count >= cap if count is not None and cap is not None else None
    patience_met = stale >= patience if stale is not None and patience is not None else None
    if not complete:
        reason, review = "incomplete_or_running", "완료 기록 없음: 실패/중단 일지와 체크포인트 확인"
    elif count is None:
        reason, review = "uncollected", "실제 반복 수/이력 미수집: 수렴 판단 불가"
    elif hit_cap:
        reason = "budget_cap"
        review = "연장 확인 후보: patience 미충족" if patience_met is False else "상한과 patience 모두 충족: 곡선 검토"
    elif patience_met:
        reason, review = "early_stopping", "patience 충족; 수렴 또는 데이터 포화의 증거는 아님"
    else:
        reason, review = "unknown_stop", "종료 근거 불충분: 기록 확인"
    return dict(iterations_run=count, best_iteration=best, cap=cap, patience=patience,
                stale_iterations=stale, hit_cap=hit_cap, patience_met=patience_met,
                stop_reason=reason, review=review)


class Evidence:
    def __init__(self, out):
        self.out, self.sources, self.issues = out, {}, []

    def read(self, path, default=None):
        path = Path(path)
        if not path.is_file():
            return default
        try:
            content = path.read_bytes()
            self.sources[os.path.relpath(path, self.out)] = hashlib.sha256(content).hexdigest()
            text = content.decode("utf-8-sig")
            if path.suffix == ".csv":
                return list(csv.DictReader(text.splitlines()))
            if path.suffix == ".jsonl":
                rows = []
                for index, line in enumerate(text.splitlines()):
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("expected JSON object")
                        rows.append(value)
                    except ValueError:
                        self.issues.append(dict(path=str(path), error=f"invalid JSONL line {index + 1}"))
                return rows
            value = json.loads(text)
            if isinstance(default, (list, dict)) and not isinstance(value, type(default)):
                raise ValueError(f"expected {type(default).__name__}, got {type(value).__name__}")
            return value
        except (OSError, ValueError) as exc:
            self.issues.append(dict(path=str(path), error=str(exc)))
            return default


def training_rows(run, cfg, evidence):
    rows, curves = [], []
    if run is None:
        return rows, curves
    opts = cfg.get("budget", {})
    for job in sorted((run / "experiment_b/models").glob("*")):
        if not job.is_dir():
            continue
        result = evidence.read(job / "complete.json", {})
        state = evidence.read(job / "training_state.json", {})
        history = evidence.read(job / "history.csv", [])
        count = result.get("epochs_run", state.get("epoch", len(history) or None))
        best = result.get("best_epoch", state.get("best_epoch"))
        row = dict(job=str(job.relative_to(run)), family="cnn", seed=result.get("seed"),
                   **budget_diagnosis(count, best, result.get("cap", opts.get("cnn_epochs")),
                       result.get("patience", opts.get("cnn_patience")), complete=bool(result)),
                   seconds=result.get("seconds_this_invocation"), timing_scope="last_training_invocation",
                   history_status="available" if history else "uncollected", metric="validation AP",
                   evidence=str(job.relative_to(run) / "history.csv"))
        rows.append(row)
        curves.append((row, history, "validation_auprc", "train_loss", "learning_rate"))
    known = set()
    for path in sorted(run.rglob("*.training.json")):
        if "visit_audit" in path.relative_to(run).parts:
            continue
        result = evidence.read(path, {})
        model = path.with_name(path.name.removesuffix(".training.json"))
        known.add(model)
        history_path = model.with_name(model.name + ".history.csv")
        history = evidence.read(history_path, [])
        row = dict(job=str(model.relative_to(run)), family=result.get("family", "tree"), seed=result.get("seed"),
                   **budget_diagnosis(result.get("iterations_run"), result.get("best_iteration"),
                                      result.get("cap"), result.get("patience"), complete=model.is_file()),
                   reported_stop_reason=result.get("stop_reason"), seconds=result.get("seconds"),
                   timing_scope="fit", history_status="available" if history else "uncollected",
                   metric=result.get("score_metric"), evidence=str(history_path.relative_to(run)))
        rows.append(row)
        curves.append((row, history, result.get("score_column", "validation/PRAUC"), "learn/Logloss", None))
    # Legacy models are not loaded/unpickled just to guess training duration or convergence.
    legacy = list(run.rglob("*.cbm")) + [p for p in run.glob("*/models/*xgb*.json")
                                       if not p.name.endswith(".training.json")]
    for model in sorted(set(legacy) - known):
        rows.append(dict(job=str(model.relative_to(run)), family="tree", seed=None,
                         **budget_diagnosis(None, None, opts.get("tree_iterations"), opts.get("tree_patience")),
                         history_status="uncollected", evidence=str(model.relative_to(run))))
    for row, history, metric, _, _ in curves:
        values = [finite(item.get(metric)) for item in history]
        if values and all(value is not None for value in values):
            window = min(20, len(values))
            before = max(values[:-window]) if len(values) > window else None
            row.update(recent_window=window, recent_best_gain=max(values[-window:]) - before if before is not None else None,
                       last_validation_score=values[-1])
    return rows, curves


def esc(value):
    return html.escape(str(value if value is not None else "미수집"))


def link(path, out, title=None):
    relative = Path(os.path.relpath(path, out)).as_posix()
    return f"<a href='{html.escape(quote(relative), quote=True)}'>{esc(title or relative)}</a>"


def tabulate(rows, columns):
    if not rows:
        return "<p>자료 없음 / 아직 실행되지 않음</p>"
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{esc(label)}</th>" for _, label in columns) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{esc(row.get(key))}</td>" for key, _ in columns) + "</tr>" for row in rows) + "</tbody></table></div>"


def bars(items, title):
    if not items:
        return "<p>" + esc(title) + ": 미수집</p>"
    height = 38 * len(items) + 35
    maximum = max(float(value) for _, value in items) or 1
    svg = [f"<svg role='img' aria-label='{esc(title)}' viewBox='0 0 900 {height}'><title>{esc(title)}</title>"]
    for index, (label, value) in enumerate(items):
        y = index * 38 + 10
        svg.append(f"<text x='0' y='{y + 20}'>{esc(label)}</text><rect x='310' y='{y}' width='{500 * float(value) / maximum:.2f}' height='26' fill='#268391'/><text x='{320 + 500 * float(value) / maximum:.2f}' y='{y + 20}'>{float(value):,.1f}</text>")
    return "".join(svg) + "</svg>"


def line_plot(history, column, title):
    pairs = [(index + 1, finite(row.get(column))) for index, row in enumerate(history)] if column else []
    pairs = [(x, y) for x, y in pairs if y is not None]
    if not pairs:
        return f"<p>{esc(title)}: 미수집</p>"
    lower, upper = min(y for _, y in pairs), max(y for _, y in pairs)
    delta = upper - lower or max(abs(upper) * .01, .001)
    points = " ".join(f"{65 + (x - 1) * 610 / max(len(history) - 1, 1):.2f},{150 - (y - lower) * 110 / delta:.2f}" for x, y in pairs)
    return (f"<figure><figcaption>{esc(title)} (독립 y축)</figcaption><svg role='img' aria-label='{esc(title)}' viewBox='0 0 740 185'>"
            f"<text x='0' y='42'>{upper:.4g}</text><text x='0' y='150'>{lower:.4g}</text>"
            f"<path d='M65 35 V150 H685' fill='none' stroke='#aaa'/><polyline points='{points}' fill='none' stroke='#176e87' stroke-width='2'/>"
            f"<text x='65' y='177'>1</text><text x='580' y='177'>epoch / iteration {len(history)}</text></svg></figure>")


def decisions(summary, training, data, feature_rows):
    links = summary.get("links", {})
    cap = [row for row in training if row.get("hit_cap") and row.get("patience_met") is False]
    unavailable = sum(row.get("history_status") == "uncollected" for row in training)
    result = [dict(priority="P0", observation=f"원천 조사 이슈 {summary.get('issues', '미수집')}; label-only {links.get('labels_without_annotation', '미수집')}",
        explanation="입력 묶음 차이 / 누락 / 계약 변종을 구별해야 함", next_action="survey/의 이슈·연결·형식 변종을 확인하고 해당 입력 어댑터만 준비",
        evidence="survey/summary.json, survey/issues.jsonl, survey/shape_variants.csv", change_one="입력 어댑터",
        required_input="기관의 파일 묶음 설명·변종 예시", cost="미측정: 조사 소요 시간 참조",
        criterion="같은 원본에서 엄격 파싱 성공률과 제외 분모 재확인; 자동 보정하지 않음"),
      dict(priority="P0", observation=f"완료 준비 단계의 기록/산모/구간: {data.get('records', '미수집')} / {data.get('mothers', '미수집')} / {data.get('segments', '미수집')}",
        explanation="원천 파일 수와 최종 선택 태아·결측 제외 후 표본 수는 다른 분모", next_action="site_inventory·exclusions·EMR 결측을 기관 담당자와 검토",
        evidence="data/summary.json, data/site_inventory.csv, data/exclusions.csv, data/emr_missingness.csv",
        change_one="표본 범위 또는 사전 정의한 품질 기준", required_input="제외 사유·원천 연결 키 확인", cost="검토 시간 미측정",
        criterion="기관/클래스별 사용 가능 산모 수, 제외 변화와 예측 시점 가용성 확인"),
      dict(priority="P1", observation=f"상한 도달 + patience 미충족 {len(cap)}개; 학습 이력 미수집 {unavailable}개",
        explanation="상한 종료는 수렴 실패의 증명도, early stopping은 수렴의 증명도 아님",
        next_action="해당 후보만 validation 기반 예산 비교를 사전 정의; 미수집 트리는 다음 실행에서 기록",
        evidence="training_diagnostics.csv, 원본 history.csv", change_one="학습 예산 (CNN cosine 일정 변경은 별도 비교)",
        required_input="보존된 체크포인트·개발 분할·학습률 일정", cost="후보별 기존 seconds/epoch_seconds 참고; 자동 외삽하지 않음",
        criterion="validation 개선량·seed 안정성·추가 비용; test로 연장 후보 선택 금지"),
      dict(priority="P1", observation="산모 수 learning curve 미실행; 데이터 양에 따른 포화는 아직 미측정",
        explanation="최적화 종료와 데이터 양 포화는 별개", next_action="train 산모의 25/50/75/100%로 고정 validation·seed 반복 비교 설계",
        evidence="미측정 — 현재 학습 곡선으로 대체 불가", change_one="train 산모 수", required_input="산모·기관·클래스 층화 가능성, 잠긴 validation",
        cost="추가 학습이 필요; 이번 기본 실행에 자동 추가하지 않음", criterion="동일 예산에서 산모 수당 validation 개선 및 클래스 부족 확인"),
      dict(priority="P0", observation="시각·장비·라벨 판독 과정·EMR 0/9999 의미 미확인",
        explanation="키/값 분포만으로 임상적 의미를 확정할 수 없음", next_action="questions.md에 담당자 답변·근거·확인 날짜를 기록",
        evidence="survey/fields.csv, questions.md", change_one="과제/변수 정의", required_input="기관 데이터 사전·판독 프로토콜",
        cost="기관 확인 필요", criterion="Birth Date를 기록 시각으로 대용하지 않음; 확인 전 시간 검증 주장 금지"),
      dict(priority="P0", observation="첫 방문 결과를 본 뒤의 설계 변경은 탐색적",
        explanation="같은 holdout 재사용은 독립 확증이 아님", next_action="다음 최종 평가용 미관측 산모/기관/기간 확보 및 내부 모델·로그 보존 승인 확인",
        evidence="run_manifest.json, splits/, events.jsonl", change_one="평가 대상 및 보존 절차",
        required_input="독립 평가 코호트·기관 반출/보존 승인", cost="기관 협의 필요",
        criterion="관측한 평가와 미관측 평가를 구분; internal 로그를 통째로 반출하지 않음")]
    problematic = [row for row in feature_rows if row.get("status") not in (None, "ok")]
    if problematic:
        result.insert(2, dict(priority="P0", observation=f"인자 품질에서 ok 아닌 항목 {len(problematic)}개",
            explanation="상수·구조적 관측 불가를 단순 0과 구분", next_action="창 길이/인자 정의와 실제 분포 대조", evidence="data/feature_quality.csv",
            change_one="인자 정의", required_input="측정 가능성 확인", cost="미측정", criterion="구조적으로 불가능한 측정을 모델 효과로 해석하지 않음"))
    return result


QUESTIONS = """# 퇴실 전 확인할 질문 (내부용)

답변, 담당자/근거 문서, 확인 날짜, 미확인 상태를 각 질문 아래에 기록하세요.
값의 분포로 의미를 추측하지 않습니다. 재생성 시 이 파일은 덮어쓰지 않습니다.

1. 파일 묶음의 모집단과 annotation/label/EMR 누락·중복 이유는?
2. 산모 ID는 기관 간에도 같은 사람을 식별하는가? 재방문·쌍태아·원 기록 연결은?
3. 실제 측정 시작/종료 시각과 장비/샘플링 정보는 어디에 있는가? Birth Date와 별개로 확인했는가?
4. FHR/TOCO 단위, 0·9999·빈칸 의미와 원본의 결측/보간/클리핑 규칙은?
5. 판독자 수, 합의·수정 이력, bbox와 분류 라벨의 버전은?
6. EMR·아웃컴은 언제 수집되며 예측 시점에 실제로 사용할 수 있는가?
7. 다음 최종 평가용 미관측 산모/기관/기간을 잠글 수 있는가? 이미 본 평가는 무엇인가?
8. 내부 데이터·체크포인트·로그 보존과 다음 방문 재개가 가능한가? 승인된 반출 범위는?

## 확인 기록

| 질문 | 답변 / 미확인 | 근거·담당자 | 확인 날짜 | 다음 조치 |
|---|---|---|---|---|
"""


def build_visit_audit(run, out, *, attempt=None):
    run, out = Path(run) if run is not None else None, Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(out)
    if attempt is None and run is not None:
        references = sorted((run / "visit_audit/attempts").glob("*.json"))
        if references:
            reference = evidence.read(references[-1], {})
            if reference.get("path"):
                attempt = (run / "visit_audit" / reference["path"]).resolve()
    attempt = Path(attempt) if attempt else None
    invocation = evidence.read(attempt / "invocation.json", {}) if attempt else {}
    manifest = evidence.read(run / "run_manifest.json", {}) if run else {}
    cfg = manifest.get("config", invocation.get("config", {}))
    summary = evidence.read(attempt / "survey/summary.json", {}) if attempt else {}
    data = evidence.read(run / "data/summary.json", {}) if run else {}
    features = evidence.read(run / "data/feature_quality.csv", []) if run else []
    training, curves = training_rows(run, cfg, evidence)
    stages = []
    if run:
        for directory in (run / ".state", run.parent / ".state"):
            for marker in sorted(directory.glob("*.json")):
                state = evidence.read(marker, {})
                stages.append(dict(stage=marker.stem, seconds=state.get("seconds"), scope="original_completed_stage"))
    events = evidence.read(attempt / "events.jsonl", []) if attempt else []
    resources = evidence.read(attempt / "resources.jsonl", []) if attempt else []
    status = evidence.read(attempt / "status.json", {"status": "running_or_uncollected"}) if attempt else {"status": "legacy_no_journal"}
    resource_summary = dict(samples=len(resources), process_peak_sampled_rss_bytes=max(
        (row["process_rss_bytes"] for row in resources if finite(row.get("process_rss_bytes")) is not None), default=None),
        minimum_sampled_disk_free_bytes=min((row["disk_free_bytes"] for row in resources if finite(row.get("disk_free_bytes")) is not None), default=None),
        gpu_observed_samples=sum(row.get("gpu_status") == "ok" for row in resources),
        gpu_scope="device_wide_not_process; sampled peaks are not true peak allocations")
    plan = decisions(summary, training, data, features)
    training_columns = ["job", "family", "seed", "iterations_run", "best_iteration", "stale_iterations", "cap", "patience",
        "hit_cap", "patience_met", "stop_reason", "review", "history_status", "metric", "recent_window", "recent_best_gain",
        "last_validation_score", "seconds", "timing_scope", "evidence", "reported_stop_reason"]
    write_csv(out / "training_diagnostics.csv", training, training_columns)
    write_csv(out / "stage_timings.csv", stages, ["stage", "seconds", "scope"])
    write_csv(out / "next_visit_plan.csv", plan, list(plan[0]))
    if not (out / "questions.md").exists():
        (out / "questions.md").write_text(QUESTIONS, encoding="utf-8")
    result = dict(generated=stamp(), privacy="internal_only_not_export_screened", invocation_status=status,
                  generator_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in ("visit_audit.py", "survey.py", "telemetry.py")},
                  source_survey_status=summary.get("status", "uncollected"), prepared_data=data,
                  training_jobs=len(training), stop_counts=dict(Counter(row["stop_reason"] for row in training)),
                  resource_summary=resource_summary, learning_curve_status="not_run",
                  evidence_errors=evidence.issues, source_files=evidence.sources)
    save_json(out / "summary.json", result)
    page = ["<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>첫 방문 진단 · 다음 방문 준비</title><style>body{background:#f4f7fa;color:#203749;font:16px/1.6 system-ui;margin:24px auto;max-width:1250px;padding:20px}section,details{background:white;padding:22px;margin:20px 0;border-radius:12px}svg{width:100%;max-height:550px}svg text{font:14px system-ui}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#edf3f4}.scroll{overflow-x:auto}a{color:#176e87}.warn{background:#fff1d5;padding:15px}figure{margin:12px 0}</style>",
        "<h1>첫 방문 진단 · 다음 방문 준비</h1><p class='warn'>내부 전용 · ID/경로/원 사이트를 포함할 수 있습니다. 반출 심사된 집계가 아닙니다.</p>",
        f"<p>생성: {esc(result['generated'])} / 이번 실행: {esc(status.get('status'))} / 실행 종류: {esc(invocation.get('mode'))} / profile: {esc(cfg.get('profile'))}</p>",
        "<p>기존 실험은 그대로 유지합니다. 여기서는 데이터의 모양·계산 비용·학습 종료 근거를 정리합니다. 학습 진단에서 test 성능은 읽지 않습니다.</p>",
        "<section><h2>1. 입력은 어디까지 연결됐나?</h2>"]
    links = summary.get("links", {})
    page.append(tabulate([dict(kind=key, files=value, parsed=summary.get("parsed_json_files", {}).get(key, "대상 외"),
                              bytes=summary.get("bytes", {}).get(key)) for key, value in summary.get("inventory", {}).items()],
                         [("kind", "원천 종류"), ("files", "파일 수"), ("parsed", "JSON 파싱 성공"), ("bytes", "bytes")]))
    page.append(bars([(label, links[key]) for key, label in [
        ("annotation_ids", "annotation 고유 ID"), ("annotation_with_label", "+ 라벨 파일 연결"),
        ("annotation_with_label_and_emr", "+ EMR 파일 연결")] if key in links], "원천 ID 파일 연결 수"))
    page.append("<p>파일 이름 기준 연결입니다. 파싱 성공·산모 연결·쌍태아 선택·분석 적격 여부를 뜻하지 않습니다. 아래의 실제 준비 결과와 분모가 다릅니다.</p>")
    page.append(f"<p>조사 이슈: {esc(summary.get('issues'))} · 다른 ID의 동일 원천 신호 표현: {esc(summary.get('exact_source_signal_duplicate_groups'))}그룹. 쌍태아 원본 공유일 수 있으므로 누수/제외로 자동 판정하지 않습니다.</p>")
    page.append(tabulate([dict(item=key, value=value) for key, value in data.items() if not isinstance(value, (list, dict))], [("item", "실제 준비 결과"), ("value", "값")]))
    if attempt:
        for filename in ("summary.json", "source_inventory.csv", "fields.csv", "shape_variants.csv", "site_quality.csv", "raw_signal_quality.csv", "duplicate_signals.csv", "duplicate_ids.csv", "issues.jsonl"):
            path = attempt / "survey" / filename
            if path.is_file():
                page.append(link(path, out) + " · ")
        sites = evidence.read(attempt / "survey/site_quality.csv", [])
        page.append("<h3>기관 코드별 원천 FHR 0 비율 (%)</h3><p>보간 전, 전체 원천 샘플 분모. 원천 중복/두 태아 포함. 기관명과 실제 결측 의미는 확인 필요.</p>")
        page.append(bars([(row["site"], 100 * float(row["fhr_zeros"]) / float(row["fhr_samples"]))
                          for row in sites if finite(row.get("fhr_samples")) and float(row["fhr_samples"]) > 0], "기관별 FHR 0 비율"))
    else:
        page.append("<p>원천 조사 미수집. 기존 분석 결과만 읽었으며 원천을 새로 조사하지 않았습니다.</p>")
    page += ["</section><section><h2>2. 학습은 왜 끝났나?</h2><p>상한 도달과 patience 충족을 별개로 표시합니다. 어느 쪽도 수렴/데이터 포화를 증명하지 않습니다. 최근 개선은 마지막 최대 20회 최고값 − 이전 최고값이며 유의성 검정이 아닙니다. CNN AP, CatBoost PRAUC, XGBoost aucpr은 구현이 달라 절대값을 직접 대조하지 않습니다.</p>",
        bars(list(result["stop_counts"].items()), "학습 종료 상태"),
        link(out / "training_diagnostics.csv", out, "모든 후보의 진단표 CSV")]
    columns = [("job", "후보"), ("iterations_run", "실제 반복"), ("best_iteration", "최고 반복"),
               ("stale_iterations", "최고점 이후"), ("cap", "상한"), ("patience", "patience"), ("review", "해석")]
    flagged = [row for row in training if row.get("hit_cap") or row["stop_reason"] in ("incomplete_or_running", "unknown_stop")]
    page += [f"<h3>우선 확인할 후보 {len(flagged)}개 (최대 20개 표시)</h3>", tabulate(flagged[:20], columns),
             f"<details><summary>전체 {len(training)}개 후보 표 펼치기</summary>", tabulate(training, columns), "</details>"]
    # All CNN curves, plus capped tree candidates: complete tree histories remain linked in CSV.
    tree_shown = 0
    for row, history, score, loss, lr in curves:
        if row["family"] != "cnn":
            if not row.get("hit_cap") or tree_shown >= 12:
                continue
            tree_shown += 1
        page += [f"<details><summary>{esc(row['job'])} · {esc(row['review'])}</summary>",
                 line_plot(history, score, row.get("metric", "validation score")),
                 line_plot(history, loss, "학습 loss (CNN: 가중 BCE)"),
                 line_plot(history, lr, "실제 epoch 시작 학습률") if lr else "<p>트리 학습률: 설정 0.05 고정; 원본 이력은 진단표 evidence 참조.</p>", "</details>"]
    page += ["</section><section><h2>3. 어디서 시간이 들었나?</h2><p>막대는 완료 단계의 원래 실행 시간입니다. 이번 재개가 같은 시간을 썼다는 의미는 아닙니다. 실패/진행 중 단계와 해시 시간은 events.jsonl을 확인하세요.</p>",
             bars([(row["stage"], row["seconds"]) for row in stages if finite(row.get("seconds")) is not None], "완료 단계 소요 시간 (초)"),
             tabulate([dict(item=key, value=value) for key, value in resource_summary.items()], [("item", "자원 관측"), ("value", "값")])]
    if attempt:
        for name in ("events.jsonl", "resources.jsonl", "console.log", "failure.json", "invocation.json"):
            if (attempt / name).is_file():
                page.append(link(attempt / name, out) + " · ")
    page += [f"<p>이번 일지 이벤트 {len(events)}개. 자원은 주기 표본이며 순간 피크를 놓칠 수 있습니다. GPU 사용률/VRAM은 장치 전체 값입니다.</p>",
             "</section><section><h2>4. 2주 동안 무엇을 준비할까?</h2>",
             tabulate(plan, [("priority", "순위"), ("observation", "관측"), ("next_action", "다음 행동"), ("criterion", "판단 기준")]),
             link(out / "next_visit_plan.csv", out, "설명·한 요소·필수 입력·비용·판단 기준 전체 CSV") + " · " + link(out / "questions.md", out, "기관 질문/답변 기록지"),
             "<p>산모 수 learning curve와 추가 예산 실험은 자동 실행하지 않았습니다. 실제 측정하지 않은 데이터 포화를 추정하지 않습니다. 다음 평가 대상은 첫 방문에서 본 holdout과 구분하세요.</p>", "</section>"]
    if evidence.issues:
        page.append("<section><h2>읽지 못한 근거 (불완전한 보고서)</h2>" + tabulate(evidence.issues, [("path", "파일"), ("error", "오류")]) + "</section>")
    page.append("</html>")
    # Include sources read during rendering too.
    save_json(out / "summary.json", result)
    temporary = out / "index.html.tmp"
    temporary.write_text("\n".join(page), encoding="utf-8")
    temporary.replace(out / "index.html")
    return out / "index.html"
