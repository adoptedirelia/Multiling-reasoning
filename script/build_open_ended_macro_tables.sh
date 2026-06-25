#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m src.evaluation.v2.open_ended_macro_table \
  --results-root "${REPO_ROOT}/results" \
  --out-dir "${REPO_ROOT}/results/open_ended_macro_table"
