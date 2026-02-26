# Error Propagation Pipeline

This pipeline supports both MKQA and Global PIQA with the same language-first flow:

1. Load dataset records per target language.
2. Build English from target-language input via MT1 (`x_l -> x_en`).
3. Build corruption instances on that English.
4. Run reasoner + MT2 cascades for evaluation.

## Core behavior

- Corruption rows always include `lang`.
- For each language, examples are independent.
- MKQA is no longer using gold `x_en`.
- Output corruptions are generated from baseline reasoner outputs, not gold English answers.

## Corruption JSONL schema

Each row contains:

- `lang`
- `example_id`
- `error_group` (`input_err` or `output_err`)
- `error_type`
- `x_en`
- `x_en_err` (input errors)
- `r_en_err`, `y_en_err` (output errors)

## Prompts and extraction

- Reasoning prompt requires:
  - `<think>` with 2-3 English sentences
  - `<answer>` final English answer
- Tag extractors use the last tag occurrence to avoid picking prompt examples.

## Run modes

### 1) Corruptions only

Build corruption JSONL without running MT2 evaluation.

```bash
python -m src.evaluation.error_prop.corruption_builder --config <config.json>
```

### 2) Full evaluation

Run baseline + input/output error cascades and compute metrics.

```bash
python -m src.evaluation.error_prop.main --config <config.json>
```

If `corruption.input_jsonl` is set, `main` reuses that file and does not rebuild corruptions.
If `corruption.input_jsonl` is null, `main` calls the builder first.

## Configs (new naming)

### MKQA

- Corruptions: `configs/error_prop_mkqa_corruptions.json`
- Eval: `configs/error_prop_mkqa_eval.json`

### Global PIQA

- Corruptions: `configs/error_prop_global_piqa_corruptions.json`
- Eval: `configs/error_prop_global_piqa_eval.json`

## Scripts (new naming)

### MKQA

- Corruptions: `src/evaluation/scripts/run_error_prop_corruptions_mkqa.sh`
- Eval: `src/evaluation/scripts/run_error_prop_eval_mkqa.sh`

### Global PIQA

- Corruptions: `src/evaluation/scripts/run_error_prop_corruptions_global_piqa.sh`
- Eval: `src/evaluation/scripts/run_error_prop_eval_global_piqa.sh`

All scripts accept override:

```bash
CONFIG_PATH=<path/to/config.json> sbatch <script.sh>
```

## Outputs

- Predictions: `<outputs.predictions_dir>/<tag>/<lang>.jsonl`
- Metrics: `<outputs.metrics_dir>/<tag>/<lang>/metrics.json`
- Logs:
  - `<outputs.logs_dir>/error_sim/<run_name>.log`
  - `<outputs.logs_dir>/error_sim/<run_name>/<lang>.jsonl`

`tag` includes:

- `baseline/standard`
- `baseline/context`
- `input_err/<error_type>/standard`
- `input_err/<error_type>/context`
- `output_err/<error_type>/standard`
- `output_err/<error_type>/context`
