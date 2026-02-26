# V2 Evaluation Pipeline

This folder contains a self-contained pipeline for:
- corruption construction
- prediction generation (standard/context/direct)
- metric evaluation (f1/bleu/bertscore/win_rate)

## Main CLIs

- Build corruptions:
  - `python -m src.evaluation.v2.corruptions.builder --config <config.json>`
- Generate predictions:
  - `python -m src.evaluation.v2.cascade.runner --config <config.json> --corruptions <path> --out <pred_dir> --modes standard,context`
  - `--modes` supports `standard`, `context`, `direct`, or `both` (expands to `standard,context`).
- Run metrics:
  - `python -m src.evaluation.v2.metrics.runner --config <config.json> --predictions <pred_dir_or_jsonl> --prediction_field <field> --methods f1,bleu,bertscore,win_rate --out <metrics.json>`
- Run wrapper pipeline:
  - `python -m src.evaluation.v2.wrappers.pipeline --config <config.json> --steps build,predict,eval`

## Prediction Row Schema

Primary v2 prediction rows contain:
- `lang`
- `example_id`
- `error_group` (`baseline`, `input_err`, `output_err`)
- `error_type` (`null` for baseline)
- `cascade_mode` (`standard`, `context`, `direct`)
- `slice` (for example `baseline/standard`)
- `prediction`

Additional fields may include `x_l`, `x_en`, `r_en`, `y_en`, and corruption fields.

For compatibility with legacy rows:
- `runner.py` can infer slice from `error_group` + `error_type` + `cascade`/`cascade_mode`.
- set `--prediction_field y_l` for legacy error-prop files.

## Win-Rate Definitions

- `context_vs_standard_win_rate`:
  - On context slices, judge picks whether context prediction beats standard prediction for the same `example_id`.
- `standard_vs_direct_win_rate`:
  - On `baseline/standard`, judge picks whether standard beats direct for the same `example_id`.
  - Requires `baseline/direct` rows in prediction input.

### Optional Judge Output Dump

You can save per-example judge decisions (the raw model output and parsed 0/1) when running `win_rate`.

Config fields (`eval` block):
- `win_write_judgments` (bool, default `false`): enable writing judge outputs JSONL.
- `win_judgments_jsonl` (string, default empty): output path for judge rows.
  - If empty, defaults to `<metrics_out_without_ext>.judge_outputs.jsonl`.

Each JSONL row includes:
- `comparison` (`context_vs_standard` or `standard_vs_direct`)
- `slice`, `lang`, `example_id`
- `question`, `gold_answers`
- `candidate_0`, `candidate_1`
- `judge_raw` (raw judge response text)
- `judge_choice` (parsed integer `0` or `1`)

## Two Ways To Include Direct

1) Unified prediction set (preferred when space allows)
- Build/merge predictions so `baseline/direct` rows are present with standard/context in the same input.
- Then run one `metrics.runner` pass.

2) Lightweight add-direct update
- Use `python -m src.evaluation.v2.metrics.add_direct_winrate ...`
- This reads existing normal preds + direct preds and injects only `standard_vs_direct_win_rate` into an existing metrics JSON.

## Merge Utility

- `python -m src.evaluation.v2.metrics.merge_direct_eval --base_dir <normal_preds_dir> --direct_dir <direct_preds_dir> --out_dir <merged_dir> --langs <csv>`

Use this when you want explicit merged prediction artifacts.

## Dataset Config Notes

`dataset.dataset_type` supports:
- `mkqa`
- `global_piqa`
- `aya`

Key fields:
- `langs`: language list to run
- `max_examples`
- `hf_name`, `hf_split`, `hf_configs` for HF datasets
- `mkqa_path` for MKQA

Aya Dolly-specific filtering in loader:
- for `dolly_machine_translated`:
  - `arb` requires `script=Arab`
  - `zho` requires `script=Hans`

## Scripts (Current Usage)

Common slurm scripts live in:
- `src/evaluation/scripts/`
- `src/evaluation/scripts/legacy/`

Typical patterns:
- prediction-only for subset
- win-rate with direct merge
- add-direct win-rate updates

Always check script defaults (`CONFIG_PATH`, `BASE_DIR`, `DIRECT_DIR`, `OUT_JSON`, `LANGS`) before running.

## Reproducibility Notes

- Keep config files versioned and immutable for a run.
- Record exact script + config + commit hash used.
- If API models are used, transient retries/backoff are bounded in `src/eval/engine/openai.py`.
