"""Build the PNG review atlas from an existing run, without re-training.

The new review bundle has its own manifest and an images/ directory containing
only PNG graphs. Existing outputs and stage hashes are never overwritten.
"""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fg.export_review import load_json, run_export_review, validate_review_bundle


def build_existing_run(path, output=None, minimum=None):
    supplied = Path(path).absolute()
    internal = supplied / "internal" if (supplied / "internal").is_dir() else supplied
    # load_json checks links and containment before reading any internal file.
    manifest = load_json(internal, "run_manifest.json")
    cfg = manifest.get("config")
    if not isinstance(cfg, dict) or cfg.get("profile") not in ("full", "mock"):
        raise ValueError("Existing internal/run_manifest.json with full/mock config is required")
    cfg = dict(cfg)
    prior_minimum = max(2, int(cfg.get("export_min_mothers", 10)))
    if minimum is not None and minimum < prior_minimum:
        raise ValueError("This command can only retain or raise the original screening minimum")
    cfg["export_min_mothers"] = minimum if minimum is not None else prior_minimum
    if "budget" not in cfg:
        cfg["budget"] = cfg.get("profiles", {}).get(cfg["profile"], {})
    out = Path(output).absolute() if output else internal.parent / "image_review"
    if out.exists():
        raise ValueError("Choose a new --output directory; existing review bundles are not overwritten")
    if out.resolve() == internal.resolve() or out.resolve().is_relative_to(internal.resolve()):
        raise ValueError("Review output must be outside internal/")
    run_export_review(internal, out, cfg)
    validate_review_bundle(out)
    return out / "images"


def main():
    parser = argparse.ArgumentParser(description="기존 결과에서 PNG 그래프 반출 심사 묶음 생성 (재학습 없음)")
    parser.add_argument("run", type=Path, help="full/mock 실행 폴더 또는 그 안의 internal 폴더")
    parser.add_argument("--output", type=Path, help="새 심사 폴더; 기본값: 실행폴더/image_review")
    parser.add_argument("--min-mothers", type=int, help="기존 선별 기준 유지 또는 상향만 가능")
    args = parser.parse_args()
    images = build_existing_run(args.run, args.output, args.min_mothers)
    print(f"PNG 그래프 심사 후보: {images}")
    print(f"이미지 {len(list(images.glob('*.png')))}개. 기관 승인 후 승인된 PNG만 반출하세요.")


if __name__ == "__main__":
    main()
