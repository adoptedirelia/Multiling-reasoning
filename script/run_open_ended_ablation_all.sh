#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

MODELS=(
  "llama"
  "mistral"
)

DATASETS=(
  "aya"
  "blend"
  "global_piqa"
  "mkqa"
)

VARIANTS=(
  "answer_plus_source_question"
  "answer_plus_english_question"
  "answer_plus_reasoning"
)

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    config_path="${REPO_ROOT}/config/${model}_${dataset}.json"
    for variant in "${VARIANTS[@]}"; do
      result_root="${REPO_ROOT}/results/${model}/${dataset}-ablation/${variant}"
      raw_dir="${result_root}/raw"
      preds_dir="${result_root}/preds"
      metrics_json="${result_root}/metrics/metrics.json"

      echo "==> ${model}/${dataset}-ablation/${variant}: raw"
      "${PYTHON_BIN}" -m src.evaluation.v2.cascade.raw_runner \
        --config "${config_path}" \
        --out "${raw_dir}" \
        --modes "${variant}" \
        --baseline-only

      echo "==> ${model}/${dataset}-ablation/${variant}: extract"
      "${PYTHON_BIN}" -m src.evaluation.v2.cascade.extract_runner \
        --raw "${raw_dir}" \
        --out "${preds_dir}"

      echo "==> ${model}/${dataset}-ablation/${variant}: metrics"
      "${PYTHON_BIN}" -m src.evaluation.v2.metrics.runner \
        --config "${config_path}" \
        --predictions "${preds_dir}" \
        --prediction_field prediction \
        --methods chrf \
        --out "${metrics_json}"
    done
  done
done
