"""Create an onsite reading guide from a completed run without retraining."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fg.common import read_json
from fg.onsite_figures import ONSITE_DIRECTORY, build_onsite_figures
from run import acquire_lock


def build_existing_run(path, name="onsite_figures"):
    supplied = Path(path).resolve()
    internal = supplied / "internal" if (supplied / "internal").is_dir() else supplied
    if internal.name != "internal" or not (internal / "run_manifest.json").is_file():
        raise ValueError("A v2 completed run or its internal/ directory is required")
    if not ONSITE_DIRECTORY.fullmatch(name):
        raise ValueError("--name must be an onsite_figures directory name (optionally with a suffix)")
    review = internal.parent / "export_review"
    manifest = review / "EXPORT_MANIFEST.json"
    # Preserve the old immutable bundle layout when adding a guide to a legacy run.
    current_layout = manifest.is_file() and read_json(manifest).get("schema_version") == 3
    out = (review if current_layout else internal) / name
    cfg = read_json(internal / "run_manifest.json")["config"]
    with acquire_lock(internal.parent):
        if read_json(internal.parent / "status.json").get("status") != "complete":
            raise ValueError("Wait for the analysis to complete before building figures")
        if out.exists() or out.is_symlink():
            raise ValueError("Output already exists; choose a new --name")
        return build_onsite_figures(internal, out, cfg)


def main():
    parser = argparse.ArgumentParser(description="기존 결과에서 현장 이해용 그래프 생성 (재학습 없음)")
    parser.add_argument("run", type=Path)
    parser.add_argument("--name", default="onsite_figures", help="New onsite_figures[_suffix] directory; export_review/ for schema 3, internal/ for older runs")
    args = parser.parse_args()
    print(build_existing_run(args.run, args.name))


if __name__ == "__main__":
    main()
