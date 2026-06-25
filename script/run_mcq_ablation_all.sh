#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset}"
MODELS="${MODELS:-llama mistral}"
DATASETS="${DATASETS:-mmlu belebele global-piqa-mc mcsqa mgsm mmath}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

VARIANTS=(
  "answer_plus_source_question"
  "answer_plus_english_question"
  "answer_plus_reasoning"
)

resolve_dataset_path() {
  local dataset_key="$1"
  local candidates=()
  case "${dataset_key}" in
    mmlu) candidates=("GlobalMMLU_test.json" "GlobalMMLU.json") ;;
    belebele) candidates=("Belebele_test.json" "Belebele.json") ;;
    global-piqa-mc) candidates=("PIQA_test.json" "PIQA.json") ;;
    mcsqa) candidates=("MCSQA_test.json" "MCSQA.json") ;;
    mgsm) candidates=("MGSM_test.json" "MGSM.json") ;;
    mmath) candidates=("MMath_test.json" "MMath.json") ;;
    *) echo "Unsupported dataset: ${dataset_key}" >&2; return 1 ;;
  esac

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -f "${DATASET_DIR}/${candidate}" ]]; then
      printf '%s\n' "${DATASET_DIR}/${candidate}"
      return 0
    fi
  done

  echo "Could not find dataset JSON for ${dataset_key} in ${DATASET_DIR}" >&2
  echo "Tried: ${candidates[*]}" >&2
  return 1
}

question_type_for() {
  local dataset_key="$1"
  case "${dataset_key}" in
    mgsm|mmath) printf 'math\n' ;;
    *) printf 'mc\n' ;;
  esac
}

baseline_type_for_variant() {
  local variant="$1"
  case "${variant}" in
    answer_plus_source_question) printf 'ablation_answer_orig_q\n' ;;
    answer_plus_english_question) printf 'ablation_answer_eng_q\n' ;;
    answer_plus_reasoning) printf 'ablation_answer_reasoning\n' ;;
    *) echo "Unsupported variant: ${variant}" >&2; return 1 ;;
  esac
}

write_config() {
  local config_path="$1"
  local model="$2"
  local dataset_path="$3"
  local output_dir="$4"
  local question_type="$5"

  local model_type=""
  local model_name=""
  local module=""
  case "${model}" in
    llama)
      model_type="llama"
      model_name="meta-llama/Llama-3.1-8B-Instruct"
      module="src.eval.main"
      ;;
    mistral)
      model_type="mistral"
      model_name="mistralai/Mistral-7B-Instruct-v0.3"
      module="src.eval.main"
      ;;
    *)
      echo "Unsupported model: ${model}" >&2
      return 1
      ;;
  esac

  cat > "${config_path}" <<JSON
{
  "mt1_config": {
    "model_type": "${model_type}",
    "model_name": "${model_name}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P}
  },
  "mt2_config": {
    "model_type": "${model_type}",
    "model_name": "${model_name}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P}
  },
  "llm_config": {
    "model_type": "${model_type}",
    "model_name": "${model_name}",
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "temperature": ${TEMPERATURE},
    "top_p": ${TOP_P}
  },
  "dataset_path": "${dataset_path}",
  "output_dir": "${output_dir}",
  "batch_size": ${BATCH_SIZE},
  "question_type": "${question_type}",
  "save_intermediate": true
}
JSON

  printf '%s\n' "${module}"
}

for model in ${MODELS}; do
  for dataset in ${DATASETS}; do
    dataset_path="$(resolve_dataset_path "${dataset}")"
    question_type="$(question_type_for "${dataset}")"

    for variant in "${VARIANTS[@]}"; do
      baseline_type="$(baseline_type_for_variant "${variant}")"
      result_root="${REPO_ROOT}/results/${model}/${dataset}-ablation/${variant}"
      raw_dir="${result_root}/raw"
      metrics_json="${result_root}/metrics/metrics.json"
      mkdir -p "${raw_dir}" "$(dirname "${metrics_json}")"

      tmp_config="$(mktemp "${TMPDIR:-/tmp}/mcq_ablation_${model}_${dataset}_${variant}.XXXXXX.json")"
      module="$(write_config "${tmp_config}" "${model}" "${dataset_path}" "${raw_dir}" "${question_type}")"

      echo "==> ${model}/${dataset}-ablation/${variant}: raw"
      "${PYTHON_BIN}" -m "${module}" \
        --config "${tmp_config}" \
        --baseline_type "${baseline_type}" \
        --output_file "${variant}.json" \
        --translation_file translations.json \
        --intermediate_file "${variant}_intermediate.json"

      echo "==> ${model}/${dataset}-ablation/${variant}: export metrics"
      "${REPO_ROOT}/script/export_pulled_accuracy_metrics.sh" \
        --dataset-key "${dataset}" \
        --model "${model}" \
        --dataset-json "${dataset_path}" \
        --input-json "${raw_dir}/${variant}.json" \
        --slice-name "baseline/${variant}" \
        --out "${metrics_json}" \
        --run-name "${model}_${dataset}_${variant}_accuracy"

      rm -f "${tmp_config}"
    done
  done
done
