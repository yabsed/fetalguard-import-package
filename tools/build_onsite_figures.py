"""Create an onsite reading guide from a completed run without retraining."""
import argparse
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fg.common import read_json
from fg.onsite_figures import run_onsite_figures
from run import acquire_lock


def build_existing_run(path, name="onsite_figures"):
    supplied = Path(path).resolve()
    internal = supplied / "internal" if (supplied / "internal").is_dir() else supplied
    if internal.name != "internal" or not (internal / "run_manifest.json").is_file():
        raise ValueError("A v2 completed run or its internal/ directory is required")
    if not name or Path(name).name != name or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError("--name must be a new directory name inside internal/")
    out = internal / name
    cfg = read_json(internal / "run_manifest.json")["config"]
    with acquire_lock(internal.parent):
        if read_json(internal.parent / "status.json").get("status") != "complete":
            raise ValueError("Wait for the analysis to complete before building figures")
        if out.exists() or out.is_symlink():
            raise ValueError("Output already exists; choose a new --name")
        # Publish only a complete guide; existing stages and their hashes stay intact.
        with tempfile.TemporaryDirectory(prefix="onsite-build-", dir=internal) as temporary:
            staging = Path(temporary)
            run_onsite_figures(internal, staging, cfg)
            staging.rename(out)
    return out / "index.html"


def main():
    parser = argparse.ArgumentParser(description="기존 결과에서 현장 이해용 그래프 생성 (재학습 없음)")
    parser.add_argument("run", type=Path)
    parser.add_argument("--name", default="onsite_figures", help="New directory name inside internal/ (default: onsite_figures)")
    args = parser.parse_args()
    print(build_existing_run(args.run, args.name))


if __name__ == "__main__":
    main()
