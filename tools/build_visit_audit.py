"""Create first-visit diagnostics from saved results without retraining."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from fg.visit_audit import build_visit_audit
from run import acquire_lock


def build_existing(path, output=None, name="visit_audit"):
    if output is None:
        from tools.build_export_diagnostics import build_existing as build_review
        return build_review(path, name)
    supplied = Path(path).resolve()
    internal = supplied / "internal" if (supplied / "internal").is_dir() else supplied
    manifest = internal / "run_manifest.json"
    if internal.name != "internal" or not manifest.is_file():
        raise ValueError("run/internal/run_manifest.json이 있는 기존 실행을 지정하세요.")
    if not re.fullmatch(r"visit_audit(?:_[a-zA-Z0-9_-]+)?", name):
        raise ValueError("--name must be visit_audit[_suffix]")
    out = Path(output).resolve() if output is not None else internal / name
    cfg = json.loads(manifest.read_text(encoding="utf-8"))["config"]
    forbidden = [PACKAGE, Path(cfg["data_root"]).resolve()]
    if any(out == parent or parent in out.parents for parent in forbidden):
        raise ValueError("Output must be outside the package and source data")
    if internal.parent in out.parents and not (out.parent == internal and re.fullmatch(r"visit_audit(?:_[a-zA-Z0-9_-]+)?", out.name)):
        raise ValueError("Inside a run, output must be internal/visit_audit[_suffix]")
    if out.exists() or out.is_symlink():
        raise ValueError("Output exists; choose a new --name or --output. Existing answers are not overwritten.")
    with acquire_lock(internal.parent):
        return build_visit_audit(internal, out)


def main():
    parser = argparse.ArgumentParser(description="저장된 이력으로 반출 검토용 방문 진단 생성: 기본 export_review/visit_audit/")
    parser.add_argument("run", type=Path)
    parser.add_argument("--name", default="visit_audit")
    parser.add_argument("--output", type=Path, help="명시할 때만 상세 내부용 진단을 지정 폴더에 생성")
    args = parser.parse_args()
    print(build_existing(args.run, args.output, args.name))


if __name__ == "__main__":
    main()
