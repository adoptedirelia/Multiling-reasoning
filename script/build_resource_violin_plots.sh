#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
OUT_DIR="${REPO_ROOT}/results/resource_violin_plots"

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/*.pdf
"${PYTHON_BIN}" -m src.evaluation.plots.resource_violin_plots \
  --output-dir "${OUT_DIR}" \
  --plots llama_mistral_global_piqa llama_mistral_mmlu_belebele
