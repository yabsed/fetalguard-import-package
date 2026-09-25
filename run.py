#!/usr/bin/env python3
"""One-command, offline pipeline. Defaults to interactive paths, never installs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbid_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError(f"Offline runtime blocked: {event}")


def config_args():
    parser = argparse.ArgumentParser(description="CTG 본선 오프라인 원클릭 분석")
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["full", "mock"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--prior-cohort", type=Path, help="이미 분석한 산모 mother_id CSV. 해당 산모는 train에만 포함")
    parser.add_argument("--check", action="store_true", help="환경·모델·데이터 디렉토리 검증만")
    parser.add_argument("--survey-only", action="store_true", help="학습·GPU 검사 없이 원천 구조/품질 조사만 (내부용)")
    parser.add_argument("--core-only", action="store_true",
                        help="사전 조사·CNN·공식 딥러닝 모델·audit/반출 산출물 없이 핵심 분석만 실행")
    args = parser.parse_args()
    if args.core_only and (args.check or args.survey_only):
        parser.error("--core-only는 --check/--survey-only와 함께 사용할 수 없습니다.")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in ("profile", "device", "threads"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key, arg, prompt in (("data_root", args.data, "데이터 폴더 (압축 해제한 상위 폴더): "),
                             ("output_root", args.output, "결과 저장 폴더 (반입 폴더 외부): ")):
        value = str(arg) if arg is not None else cfg.get(key, "")
        if not value:
            if not sys.stdin.isatty():
                parser.error(f"--{'data' if key == 'data_root' else 'output'} 경로를 지정하거나 START.ipynb의 입력창을 사용하세요.")
            value = input(prompt).strip().strip('"')
        if not value:
            parser.error(f"Empty {key}")
        cfg[key] = str(Path(value).expanduser().resolve())
    data, output = Path(cfg["data_root"]), Path(cfg["output_root"])
    if not data.is_dir():
        parser.error(f"데이터 폴더 없음: {data}")
    if output == PACKAGE or PACKAGE in output.parents:
        parser.error("결과 폴더는 반입 패키지 밖에 두세요.")
    if output == data or data in output.parents:
        parser.error("결과 폴더는 원천 데이터 밖에 두세요.")
    if cfg["threads"] < 1:
        parser.error("threads must be positive")
    if not isinstance(cfg.get("export_min_mothers", 10), int) or cfg.get("export_min_mothers", 10) < 2:
        parser.error("export_min_mothers must be an integer >=2 (screening rule, not approval)")
    prior = str(args.prior_cohort) if args.prior_cohort is not None else cfg.get("prior_cohort_file", "")
    cfg["prior_cohort_file"] = str(Path(prior).expanduser().resolve()) if prior else ""
    if prior:
        if not Path(cfg["prior_cohort_file"]).is_file():
            parser.error("Prior cohort CSV does not exist")
        cfg["prior_cohort_sha256"] = file_sha256(cfg["prior_cohort_file"])
    from fg.protocol import DESIGN_VERSION
    if args.core_only:
        cfg.update(core_only=True, audit=False, cnn=False, official_models=False)
    else:
        cfg.update(core_only=False, audit=True)
    cfg["design_version"] = DESIGN_VERSION
    cfg["budget"] = cfg["profiles"][cfg["profile"]]
    return args, cfg


def verify_package():
    manifest = PACKAGE / "PACKAGE_MANIFEST.json"
    if not manifest.is_file():
        raise ValueError("PACKAGE_MANIFEST.json 없음. 반입용 패키지 구성이 완료되지 않았습니다.")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    bad = [name for name, digest in entries["files"].items()
           if not (PACKAGE / name).is_file() or file_sha256(PACKAGE / name) != digest]
    if bad:
        raise ValueError(f"반입 파일 무결성 불일치: {bad}. config.json만 현장 설정용으로 수정할 수 있습니다.")
    return file_sha256(manifest)


def preflight(cfg):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 이상이 필요합니다. 신청한 PyTorch 커널을 선택하세요.")
    mods = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn", "catboost": "catboost",
            "xgboost": "xgboost", "matplotlib": "matplotlib", "pillow": "PIL", "tabulate": "tabulate"}
    if cfg["cnn"] or cfg["official_models"]:
        mods["torch"] = "torch"
    if cfg["official_models"]:
        mods["opencv"] = "cv2"
    versions, missing = {}, []
    for name, module in mods.items():
        try:
            loaded = importlib.import_module(module)
            versions[name] = getattr(loaded, "__version__", "unknown")
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise RuntimeError("환경 사전검사 실패. 자동 설치하지 않습니다:\n" + "\n".join(missing))
    torch = None
    if cfg["cnn"] or cfg["official_models"]:
        import torch
        import numpy as np
        torch.set_num_threads(cfg["threads"])
        torch.from_numpy(np.zeros((2, 3), np.float32)).numpy()
        if cfg["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. config device=cpu 또는 auto로 변경하세요.")
        cfg["resolved_device"] = "cuda" if cfg["device"] == "cuda" or (cfg["device"] == "auto" and torch.cuda.is_available()) else "cpu"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        cfg["resolved_device"] = "cpu"
    if cfg["official_models"]:
        from fg.official import preflight_models
        preflight_models(PACKAGE, cfg)
    return dict(python=sys.version, platform=platform.platform(), packages=versions, device=cfg["resolved_device"],
                cuda=torch.version.cuda if torch is not None else None,
                gpu=torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None)


def run_stage(run, name, action, *, target=None):
    from fg.telemetry import scope
    with scope(name):
        return _run_stage(run, name, action, target=target)


def _run_stage(run, name, action, *, target=None):
    from fg.common import sha256, write_json, read_json, note
    from fg.telemetry import emit
    from fg.resilience import ArtifactIntegrityError
    target = Path(target) if target is not None else run / name
    marker = run / ".state" / (name + ".json")
    if marker.exists():
        completed = read_json(marker)
        if all((target / p).is_file() and sha256(target / p) == digest for p, digest in completed["files"].items()):
            note(f"재개: {name} 완료 검증됨")
            emit("stage_reused", original_seconds=completed.get("seconds"))
            return completed.get("outcome", {"status": "complete"})
        raise ArtifactIntegrityError(f"완료 단계 산출물이 변경/삭제됨: {target}. 원본을 복구하거나 다른 --output 경로로 재실행하세요.")
    note(f"시작: {name}")
    started = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    action(target)
    hash_started = time.monotonic()
    files = {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
    if not files:
        raise RuntimeError(f"Stage produced no artifacts: {name}")
    issue_file = target / "issues_summary.json"
    if name == "onsite_figures":
        issue_file = run / "internal/onsite_issues/issues_summary.json"
    outcome = read_json(issue_file) if issue_file.exists() else {"status": "complete"}
    write_json(marker, {"seconds": time.monotonic() - started, "files": files, "outcome": outcome})
    emit("stage_artifacts_hashed", seconds=time.monotonic() - hash_started, files=len(files))
    note(f"완료: {name} ({time.monotonic() - started:.1f}s; {outcome['status']})")
    return outcome


def acquire_lock(run):
    """OS lock is released on crash too; stale PID files do not block resumption."""
    stream = (run / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f"같은 실행이 이미 진행 중입니다: {run}") from None
    return stream


def main():
    args, cfg = config_args()
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS=str(cfg["threads"]), MKL_NUM_THREADS=str(cfg["threads"]),
                      OPENBLAS_NUM_THREADS=str(cfg["threads"]), MPLCONFIGDIR=str(output / ".matplotlib"),
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    from fg.telemetry import Visit
    mode = ("survey_only" if getattr(args, "survey_only", False) else "check_only" if args.check else
            "core_only" if cfg.get("core_only") else "analysis")
    with Visit(output, cfg, mode=mode) as visit:
        return execute(args, cfg, visit)


def execute(args, cfg, visit):
    from fg.telemetry import save_json as write_json, scope, emit
    from fg.survey import run_survey
    def note(message):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)
    output = Path(cfg["output_root"])
    with scope("package_verification"):
        bundle_hash = verify_package()
    emit("package_verified", package_sha256=bundle_hash)
    if cfg.get("core_only"):
        note("핵심 분석 모드: 사전 조사·CNN·공식 딥러닝 모델·audit/반출 산출물 생략")
    else:
        note(f"학습 전 데이터 조사 · 실행 일지: {visit.path}")
        with scope("source_survey"):
            run_survey(Path(cfg["data_root"]), visit.path / "survey")
        visit.refresh()
    if getattr(args, "survey_only", False):
        note(f"SURVEY OK — 학습 없음. 조사 보고서: {visit.path / 'index.html'}")
        return
    note(f"환경·공식 모델 사전검사 ({cfg['profile']})")
    with scope("preflight"):
        environment = preflight(cfg)
    # Imports above may query the platform via subprocess; scientific runtime below is offline.
    sys.addaudithook(forbid_network)
    from fg.data import discover
    note("원천 데이터 목록·중복·해시 확인")
    with scope("input_hash_and_discovery"):
        catalog, fingerprints, duplicates = discover(Path(cfg["data_root"]))
    identity = dict(config=cfg, package_sha256=bundle_hash, input_files=fingerprints, environment=environment)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    run_root = output / f"{cfg['profile']}-{key}"
    run_root.mkdir(exist_ok=True)
    visit.run_lock = acquire_lock(run_root)
    run = run_root / "internal"
    run.mkdir(exist_ok=True)
    visit.attach(run)
    write_json(run / "run_manifest.json", dict(**identity, run_id=key, duplicate_files=duplicates))
    from fg.protocol import write_protocol
    write_protocol(run, cfg)
    visit.refresh()
    note(f"내부 결과: {run}")
    if args.check:
        note("CHECK OK — 학습은 실행하지 않았습니다.")
        return
    latest = {"run": str(run_root), "internal": str(run), "report": str(run / "report/report.html")}
    if cfg.get("audit", True):
        latest.update(export_review=str(run_root / "export_review"),
            onsite_figures=str(run_root / "export_review/onsite_figures/index.html"),
            visit_audit=str(run_root / "export_review/visit_audit/index.html"),
            internal_visit_audit=str(run / "visit_audit/index.html"),
            export_report=str(run_root / "export_review/report.html"),
            export_images=str(run_root / "export_review/images"), export_status="pending_institution_review")
    else:
        latest.update(export_review=None, onsite_figures=None, visit_audit=None,
                      internal_visit_audit=None, export_report=None, export_images=None,
                      export_status="disabled_by_core_only")
    write_json(output / "LATEST.json", latest)
    try:
        execute_analysis(run, run_root, cfg, catalog, visit)
    except BaseException as exc:
        # Explicit interrupt, integrity, permission, or storage failure is fatal.
        # Try to leave a truthful status; unavailable storage cannot be repaired here.
        try:
            write_json(run / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                              "traceback": traceback.format_exc()})
            write_json(run_root / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                  "type": type(exc).__name__, "details": "internal/failure.json"})
        except OSError:
            pass
        raise


def execute_analysis(run, run_root, cfg, catalog, visit):
    """Continue independent stages; never call a failed prerequisite usable."""
    from fg.resilience import Issues, partial_report
    from fg.telemetry import save_json
    issues = Issues(run / "pipeline_issues")
    stages = {}
    status_path = run_root / "status.json"

    def snapshot(final=False):
        state = ("complete_with_issues" if issues.rows or any(
            row["status"] != "complete" for row in stages.values()) else "complete") if final else "running"
        value = dict(status=state, stages=stages, profile=cfg["profile"], core_only=cfg.get("core_only", False),
                     design_version=cfg["design_version"], report=str(run / "report/report.html"),
                     export_review=str(run_root / "export_review") if cfg.get("audit", True) else None,
                     export_status=("pending_institution_review" if (run_root / "export_review/EXPORT_MANIFEST.json").is_file()
                                    else "unavailable") if cfg.get("audit", True) else "disabled_by_core_only",
                     finished=datetime.now(timezone.utc).isoformat() if final else None)
        save_json(run / "pipeline_status.json", value)
        # Public status contains fixed stage names/status only; exceptions stay internal.
        save_json(status_path, value)
        if final:
            visit.analysis_status = state
        return state

    def call(module, function, *values):
        return getattr(importlib.import_module(module), function)(*values)

    def attempt(name, module, function, values, dependencies=(), owner=None, target=None):
        unavailable = [dep for dep in dependencies if stages.get(dep, {}).get("status") not in {"complete", "complete_with_issues"}]
        if unavailable:
            stages[name] = dict(status="unavailable", blocked_by=unavailable)
            print(f"미산출: {name} (필수 단계 미완료: {', '.join(unavailable)}); 다음 작업 계속", flush=True)
            snapshot()
            return
        stages[name] = {"status": "running"}
        snapshot()
        before = len(issues.rows)
        with issues.guard(name):
            result = run_stage(owner or run, name, lambda out: call(module, function, *values, out, cfg), target=target)
            stages[name] = result if isinstance(result, dict) else {"status": "complete"}
        if len(issues.rows) != before:
            stages[name] = {"status": "unavailable" if issues.rows[-1]["status"] == "unavailable" else "failed",
                            "details": "internal/pipeline_issues/issues.jsonl"}
        snapshot()

    def skip(name, reason):
        target = run / name
        target.mkdir(parents=True, exist_ok=True)
        save_json(target / "status.json", {"status": reason})
        stages[name] = {"status": reason}
        snapshot()

    snapshot()
    # Data's signature is catalog, out, cfg. Later stages share run, out, cfg.
    attempt("data", "fg.data", "prepare", (catalog,))
    attempt("splits", "fg.evaluation", "create_splits", (run / "data",), ("data",))
    attempt("experiment_a", "fg.models", "run_a", (run,), ("splits",))
    if cfg.get("core_only"):
        skip("experiment_b", "skipped_by_core_only")
    else:
        attempt("experiment_b", "fg.cnn", "run_b", (run,), ("splits",))
    # Supplementary modules decide independently whether they need splits/models.
    attempt("supplementary", "fg.supplementary", "run_supplementary", (run,), ("data",))
    # official wrapper retains its public API but supports split/model-free replay.
    if cfg.get("core_only"):
        skip("official", "skipped_by_core_only")
    else:
        attempt("official", "fg.official", "run_reference", (run, catalog), ("data",))
    attempt("report", "fg.report", "run_report", (run,))
    if stages["report"]["status"] not in {"complete", "complete_with_issues"}:
        with issues.guard("partial_report"):
            partial_report(run, run / "report", cfg)
    visit.refresh()
    if cfg.get("core_only"):
        stages["export_review"] = {"status": "skipped_by_core_only"}
        stages["onsite_figures"] = {"status": "skipped_by_core_only"}
    else:
        attempt("export_review", "fg.export_review", "run_export_review", (run,), owner=run_root)
        if stages["export_review"]["status"] in {"complete", "complete_with_issues"}:
            attempt("onsite_figures", "fg.onsite_figures", "build_onsite_figures", (run,), owner=run_root,
                    target=run_root / "export_review/onsite_figures")
        else:
            stages["onsite_figures"] = dict(status="unavailable", blocked_by=["export_review"])
    state = snapshot(final=True)
    if stages["report"]["status"] not in {"complete", "complete_with_issues"}:
        with issues.guard("partial_report_final"):
            partial_report(run, run / "report", cfg)
        state = snapshot(final=True)
    issues.finish()
    prefix = "CORE ANALYSIS COMPLETE" if cfg.get("core_only") else state.upper()
    print(f"{prefix} — 실행 종료. 내부 보고서: {run / 'report/report.html'}", flush=True)
    print("미산출·제외 내역: internal/pipeline_status.json 및 각 단계 issues.jsonl. 반출은 승인된 집계만.", flush=True)


if __name__ == "__main__":
    main()
