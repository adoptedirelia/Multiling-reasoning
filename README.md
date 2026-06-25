# Multilingual Reasoning Cascades Need More Context

This repository contains the code and paper artifacts for **Multilingual Reasoning Cascades Need More Context**.

The main reproduced paper outputs in this repo are:
- open-ended evaluation runs for `llama`, `mistral`, and `gpt`
- multiple-choice and math evaluation runs
- ablation runs
- macro tables, significance tables, and resource-tier violin plots
- MT1 translation-quality analysis and agreement tables

## Repository Layout

- [config](config): paper-facing configs for the 4 open-ended datasets
- [script](script): top-level reproduction scripts
- [results](results): paper-facing outputs and tables
- [src/evaluation/v2](src/evaluation/v2): open-ended cascade and evaluation pipeline
- [src/eval](src/eval): MCQ/math evaluation pipeline
- [src/evaluation/export_accuracy_metrics.py](src/evaluation/export_accuracy_metrics.py): exporter from MCQ/math raw outputs to canonical `metrics.json`
- [src/evaluation/plots](src/evaluation/plots): plotting code used for paper figures
- [src/evaluation/sig_testing](src/evaluation/sig_testing): significance-table builders
- [src/evaluation/translation_audit](src/evaluation/translation_audit): MT1 audit code and table builders

## Open-Ended Runs

To reproduce the paper-facing open-ended runs for `aya`, `blend`, `global_piqa`, and `mkqa`:

```bash
script/run_open_ended_final_all.sh
```

This writes raw outputs, extracted predictions, and `metrics.json` files under:

```bash
results/<model>/<dataset>/
```

Models covered by the script:
- `llama` = `Llama-3.1-8B-Instruct`
- `mistral` = `Mistral-7B-Instruct-v0.3`
- `gpt` = `GPT-4o-mini`

## MCQ And Math

For the 6 MCQ/math datasets, the repo includes:
- a raw runner wrapper: [script/run_mcq_math_all.sh](script/run_mcq_math_all.sh)
- an accuracy exporter: [script/export_pulled_accuracy_metrics.sh](script/export_pulled_accuracy_metrics.sh)

The wrapper expects local benchmark JSONs in `DATASET_DIR`, for example:
- `GlobalMMLU_test.json`
- `Belebele_test.json`
- `PIQA_test.json`
- `MCSQA_test.json`
- `MGSM_test.json`
- `MMath_test.json`

Example:

```bash
DATASET_DIR=/path/to/dataset script/run_mcq_math_all.sh
```

This writes canonical accuracy metrics into:

```bash
results/<model>/<dataset>/metrics/metrics.json
```

## MCQ And Open-Ended Ablations

To reproduce the open-ended ablation runs:

```bash
script/run_open_ended_ablation_all.sh
```

To reproduce the MCQ/math ablation runs:

```bash
DATASET_DIR=/path/to/dataset script/run_mcq_ablation_all.sh
```

These populate:

```bash
results/<model>/<dataset>-ablation/
```

For MCQ/math, the variant-specific metrics are written under:

```bash
results/<model>/<dataset>-ablation/<variant>/
```

## Figures And Tables

After both the open-ended runs and the MCQ/math runs are available in `results/`, the table and figure builders consume those metrics directly from `results/`.

Open-ended macro table:

```bash
script/build_open_ended_macro_tables.sh
```

Ablation average tables:

```bash
script/build_ablation_average_tables.sh
```

Open-ended ablation figure:

```bash
script/build_open_ended_ablation_plot.sh
```

Significance tables:

```bash
script/build_significance_tables.sh
```

Resource-tier violin plots:

```bash
script/build_resource_violin_plots.sh
```

These scripts write into:
- [results/open_ended_macro_table](results/open_ended_macro_table)
- [results/ablation](results/ablation)
- [results/significance](results/significance)
- [results/resource_violin_plots](results/resource_violin_plots)

## Translation Quality Analysis

Full LLM judgments:

```bash
EXPORTS_DIR=/path/to/mt1_translations script/build_translation_quality_full_llm_judgements.sh
```

Human-annotated subset judgments:

```bash
EXPORTS_DIR=/path/to/mt1_translations script/build_translation_quality_small_llm_judgements.sh
```

Merged error-distribution table:

```bash
script/build_translation_quality_analysis.sh
```

Agreement table:

```bash
script/build_translation_quality_agreement.sh
```

Outputs live under:
- [results/translation_quality_analysis/full_llm_judgements](results/translation_quality_analysis/full_llm_judgements)
- [results/translation_quality_analysis/with_human_judge](results/translation_quality_analysis/with_human_judge)
- [results/translation_quality_analysis/tables](results/translation_quality_analysis/tables)

## Notes

- `results/` is the paper-facing output directory.
- `config/` and `script/` are the main entry points for reproduction.
