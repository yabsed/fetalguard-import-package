#!/usr/bin/env bash
set -euo pipefail
task_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_dir="$(cd -- "$task_package_dir/../.." && pwd)"
exec "${FETALGUARD_PYTHON:-python}" -B "$task_package_dir/run.py" --profile mock \
  --data "$task_repo_dir/dataset/korean-ctg/dataset" \
  --output "$task_repo_dir/.scratch/import-mock-results" "$@"
