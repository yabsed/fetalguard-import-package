"""Maintainer command: hash approved contents and optionally build an import ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = []
    allowed_json = {"config.json", "SOURCE_MANIFEST.json", "models/export_verification.json", "models/official_xgboost/xgb_clf.json"}
    for path in sorted(root.rglob("*")):
        if any(p in {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"} for p in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink forbidden: {path}")
        if not path.is_file() or path.name == "PACKAGE_MANIFEST.json":
            continue
        if path.suffix in {".csv", ".npy", ".npz", ".dat", ".hea", ".zip", ".cbm"}:
            raise ValueError(f"Unexpected dataset/derived artifact in import bundle: {path}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".parquet", ".xlsx"}:
            raise ValueError(f"Unexpected data/image in import bundle: {path}")
        if path.suffix == ".json" and str(path.relative_to(root)) not in allowed_json:
            raise ValueError(f"Unapproved JSON (possible patient record): {path}")
        files.append(path)
    manifest = {"format": 1, "data_included": False, "mutable_config": "config.json", "files": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name != "config.json"}}
    (root / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files.append(root / "PACKAGE_MANIFEST.json")
    if args.zip:
        archive = args.zip.resolve()
        if archive == root or root in archive.parents:
            raise ValueError("ZIP destination must be outside import folder")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, Path(root.name) / p.relative_to(root))
        print(f"ZIP: {archive} ({archive.stat().st_size:,} bytes)")
    print(f"Sealed {len(files)} files, {sum(p.stat().st_size for p in files):,} bytes. No dataset.")


if __name__ == "__main__":
    main()
