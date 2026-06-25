#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
OUT_DIR="${REPO_ROOT}/results/translation_quality_analysis/full_llm_judgements"
EXPORTS_DIR="${EXPORTS_DIR:-}"

if [[ -z "${EXPORTS_DIR}" ]]; then
  echo "Set EXPORTS_DIR to the MT1 translation exports directory." >&2
  exit 1
fi

rm -f "${OUT_DIR}/manifest.json"

"${PYTHON_BIN}" -m src.evaluation.translation_audit.run_mt1_translation_audit \
  --repo-root "${REPO_ROOT}" \
  --exports-dir "${EXPORTS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --judge-model "${JUDGE_MODEL:-gpt-5.4-mini}" \
  --api-mode "${OPENAI_API_MODE:-responses}" \
  --timeout-s "${TIMEOUT_S:-300}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --sample-seed "${SAMPLE_SEED:-0}" \
  --samples-per-language "${SAMPLES_PER_LANGUAGE:-30}"

rm -f "${OUT_DIR}/manifest.json"
