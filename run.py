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
    args = parser.parse_args()
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
        from fg.common import sha256
        if not Path(cfg["prior_cohort_file"]).is_file():
            parser.error("Prior cohort CSV does not exist")
        cfg["prior_cohort_sha256"] = sha256(cfg["prior_cohort_file"])
    from fg.protocol import DESIGN_VERSION
    cfg["design_version"] = DESIGN_VERSION
    cfg["budget"] = cfg["profiles"][cfg["profile"]]
    return args, cfg


def verify_package():
    manifest = PACKAGE / "PACKAGE_MANIFEST.json"
    if not manifest.is_file():
        raise ValueError("PACKAGE_MANIFEST.json 없음. 반입용 패키지 구성이 완료되지 않았습니다.")
    from fg.common import sha256
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    bad = [name for name, digest in entries["files"].items()
           if not (PACKAGE / name).is_file() or sha256(PACKAGE / name) != digest]
    if bad:
        raise ValueError(f"반입 파일 무결성 불일치: {bad}. config.json만 현장 설정용으로 수정할 수 있습니다.")
    return sha256(manifest)


def preflight(cfg):
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 이상이 필요합니다. 신청한 PyTorch 커널을 선택하세요.")
    mods = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn", "catboost": "catboost",
            "xgboost": "xgboost", "torch": "torch", "matplotlib": "matplotlib", "pillow": "PIL", "opencv": "cv2", "tabulate": "tabulate"}
    versions, missing = {}, []
    for name, module in mods.items():
        try:
            loaded = importlib.import_module(module)
            versions[name] = getattr(loaded, "__version__", "unknown")
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise RuntimeError("환경 사전검사 실패. 자동 설치하지 않습니다:\n" + "\n".join(missing))
    import torch
    import numpy as np
    torch.set_num_threads(cfg["threads"])
    torch.from_numpy(np.zeros((2, 3), np.float32)).numpy()
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. config device=cpu 또는 auto로 변경하세요.")
    cfg["resolved_device"] = "cuda" if cfg["device"] == "cuda" or (cfg["device"] == "auto" and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if cfg["official_models"]:
        from fg.official import preflight_models
        preflight_models(PACKAGE, cfg)
    return dict(python=sys.version, platform=platform.platform(), packages=versions, device=cfg["resolved_device"],
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def run_stage(run, name, action):
    from fg.common import sha256, write_json, read_json, note
    target = run / name
    marker = run / ".state" / (name + ".json")
    if marker.exists():
        completed = read_json(marker)
        if all((target / p).is_file() and sha256(target / p) == digest for p, digest in completed["files"].items()):
            note(f"재개: {name} 완료 검증됨")
            return
        raise ValueError(f"완료 단계 산출물이 변경/삭제됨: {target}. 원본을 복구하거나 다른 --output 경로로 재실행하세요.")
    note(f"시작: {name}")
    started = time.monotonic()
    target.mkdir(parents=True, exist_ok=True)
    action(target)
    files = {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
    if not files:
        raise RuntimeError(f"Stage produced no artifacts: {name}")
    write_json(marker, {"seconds": time.monotonic() - started, "files": files})
    note(f"완료: {name} ({time.monotonic() - started:.1f}s)")


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
    from fg.common import note, write_json
    bundle_hash = verify_package()
    note(f"환경·공식 모델 사전검사 ({cfg['profile']})")
    environment = preflight(cfg)
    # Imports above may query the platform via subprocess; scientific runtime below is offline.
    sys.addaudithook(forbid_network)
    from fg.data import discover, prepare
    note("원천 데이터 목록·중복·해시 확인")
    catalog, fingerprints, duplicates = discover(Path(cfg["data_root"]))
    identity = dict(config=cfg, package_sha256=bundle_hash, input_files=fingerprints, environment=environment)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    run_root = output / f"{cfg['profile']}-{key}"
    run_root.mkdir(exist_ok=True)
    run_lock = acquire_lock(run_root)
    run = run_root / "internal"
    run.mkdir(exist_ok=True)
    write_json(run / "run_manifest.json", dict(**identity, run_id=key, duplicate_files=duplicates))
    from fg.protocol import write_protocol
    write_protocol(run, cfg)
    note(f"내부 결과: {run}")
    if args.check:
        note("CHECK OK — 학습은 실행하지 않았습니다.")
        run_lock.close()
        return
    write_json(output / "LATEST.json", {"run": str(run_root), "internal": str(run),
        "report": str(run / "report/report.html"), "export_review": str(run_root / "export_review"),
        "onsite_figures": str(run / "onsite_figures/index.html"),
        "export_report": str(run_root / "export_review/report.html"),
        "export_images": str(run_root / "export_review/images"), "export_status": "pending_institution_review"})
    from fg.evaluation import create_splits
    from fg.models import run_a
    from fg.cnn import run_b
    from fg.supplementary import run_supplementary
    from fg.official import run_official
    from fg.report import run_report
    from fg.onsite_figures import run_onsite_figures
    from fg.export_review import run_export_review
    status = run_root / "status.json"
    write_json(status, {"status": "running", "profile": cfg["profile"], "started": datetime.now(timezone.utc).isoformat()})
    try:
        run_stage(run, "data", lambda out: prepare(catalog, out, cfg))
        run_stage(run, "splits", lambda out: create_splits(run / "data", out, cfg))
        run_stage(run, "experiment_a", lambda out: run_a(run, out, cfg))
        run_stage(run, "experiment_b", lambda out: run_b(run, out, cfg))
        run_stage(run, "supplementary", lambda out: run_supplementary(run, out, cfg))
        run_stage(run, "official", lambda out: run_official(PACKAGE, run, out, cfg, catalog))
        run_stage(run, "report", lambda out: run_report(run, out, cfg))
        run_stage(run, "onsite_figures", lambda out: run_onsite_figures(run, out, cfg))
        run_stage(run_root, "export_review", lambda out: run_export_review(run, out, cfg))
    except BaseException as exc:
        write_json(run / "failure.json", {"status": "failed", "error": str(exc),
                   "type": type(exc).__name__, "traceback": traceback.format_exc()})
        write_json(status, {"status": "failed", "type": type(exc).__name__,
                            "details": "internal/failure.json"})
        run_lock.close()
        raise
    write_json(status, {"status": "complete", "profile": cfg["profile"], "design_version": cfg["design_version"],
                        "report": str(run / "report/report.html"), "export_review": str(run_root / "export_review"),
                        "onsite_figures": str(run / "onsite_figures/index.html"),
                        "export_status": "pending_institution_review",
                        "finished": datetime.now(timezone.utc).isoformat()})
    note(f"SUCCESS — 현장 보고서: {run / 'report/report.html'}")
    note(f"현장 이해용 그래프: {run / 'onsite_figures/index.html'}")
    note(f"반출 심사용 집계 결과 (승인 전): {run_root / 'export_review/report.html'}")
    note(f"이미지 전용 심사 폴더 (PNG만): {run_root / 'export_review/images'}")
    run_lock.close()


if __name__ == "__main__":
    main()
