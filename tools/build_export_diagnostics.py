"""Add screened visit/data/training diagnostics to an existing export_review."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fg.export_diagnostics import build_export_diagnostics, safe_path
from run import acquire_lock


def build_existing(path, name="visit_audit"):
    supplied = safe_path(path).absolute()
    internal = supplied / "internal" if (supplied / "internal").is_dir() else supplied
    if internal.name != "internal" or not (internal / "run_manifest.json").is_file():
        raise ValueError("Existing run/internal/run_manifest.json required")
    with acquire_lock(internal.parent):
        return build_export_diagnostics(internal, internal.parent / "export_review" / name)


def main():
    parser = argparse.ArgumentParser(description="기존 내부 결과의 반출용 집계·학습곡선·방문 진단 생성 (재학습 없음)")
    parser.add_argument("run", type=Path)
    parser.add_argument("--name", default="visit_audit", help="visit_audit 또는 visit_audit_접미어")
    args = parser.parse_args()
    print(build_existing(args.run, args.name))


if __name__ == "__main__":
    main()
