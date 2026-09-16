#!/usr/bin/env bash
set -euo pipefail
task_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FETALGUARD_PYTHON:-python}" -B "$task_package_dir/run.py" "$@"
