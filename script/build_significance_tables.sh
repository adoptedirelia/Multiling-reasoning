#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m src.evaluation.sig_testing.build_ctx_std_tables \
  --repo-root "${REPO_ROOT}" \
  --out-dir "${REPO_ROOT}/results/significance"
