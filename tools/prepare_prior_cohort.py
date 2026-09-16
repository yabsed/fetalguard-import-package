"""Prepare a sensitive, site-local list of previously analyzed mothers.

Do not put this list into the import package or export-review directory.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="기존 실험에 사용한 산모를 train-only 목록으로 준비")
    parser.add_argument("previous_run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.previous_run.resolve()
    if (source / "internal").is_dir():
        source = source / "internal"
    records = source / "data/records.csv"
    package = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if destination == package or package in destination.parents or "export_review" in destination.parts:
        parser.error("산모 목록은 반입 패키지/반출 심사 묶음 밖의 승인된 내부 경로에 보관하세요.")
    if destination.exists():
        parser.error("출력 파일이 이미 있습니다. 기존 목록을 덮어쓰지 않습니다.")
    frame = pd.read_csv(records, dtype={"mother_id": str}, keep_default_na=False)
    if "mother_id" not in frame or frame.mother_id.str.strip().isin(["", "nan", "None", "9999"]).any():
        parser.error("기존 결과에 유효한 mother_id가 필요합니다.")
    mothers = pd.DataFrame({"mother_id": sorted(set(frame.mother_id.str.strip()))})
    destination.parent.mkdir(parents=True, exist_ok=True)
    mothers.to_csv(destination, index=False)
    print(f"내부 전용 산모 목록: {destination} ({len(mothers):,} mothers). 시설 사용 허가/ID 호환성을 확인하세요.")


if __name__ == "__main__":
    main()
