# Multilingual Reasoning Cascades Need More Context

This repository contains the code and paper-facing artifacts for **Multilingual Reasoning Cascades Need More Context** ([paper](https://arxiv.org/abs/2606.27306)).

Translation cascades for reasoning translate a query into English, reason in English, and translate the answer back to the original language, but this pipeline is structurally lossy because intermediate stages discard information later modules may still need. We study a simple, training-free **context-aware** cascade that preserves additional upstream context for the final translation module, and find especially strong gains on open-ended multilingual reasoning, with the original-language question carrying most of the useful context.

This repository includes code for:

- **Open-ended cascade evaluation** on `aya`, `blend`, `global_piqa`, and `mkqa`
- **Multiple-choice and math evaluation** on `mmlu`, `belebele`, `global-piqa-mc`, `mcsqa`, `mgsm`, and `mmath`
- **Ablation studies** for both open-ended and MCQ/math settings
- **Table and figure generation**, including macro tables, significance tables, and resource-tier violin plots
- **Translation-quality analysis** for MT1 outputs, including LLM-judged error distributions and agreement tables

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment and Inputs](#environment-and-inputs)
- [Open-Ended Evaluation](#open-ended-evaluation)
- [MCQ and Math Evaluation](#mcq-and-math-evaluation)
- [Ablations](#ablations)
- [Figures and Tables](#figures-and-tables)
- [Translation Quality Analysis](#translation-quality-analysis)

---

## Repository Structure

- `config/`  
  Paper-facing configs for the open-ended runs (`llama`, `mistral`, `gpt` across the 4 open-ended datasets).

- `script/`  
  Top-level reproduction scripts for evaluation, ablations, table building, plotting, and translation-quality analysis.

- `results/`  
  Paper-facing outputs. This is the directory where reproduced metrics, tables, plots, and audit artifacts are written.

- `src/evaluation/v2/`  
  Open-ended cascade pipeline, including raw generation, extraction, and metric computation.

- `src/eval/`  
  Multiple-choice and math evaluation pipeline.

- `src/evaluation/export_accuracy_metrics.py`  
  Exporter that converts MCQ/math raw outputs into the canonical `metrics.json` format used by downstream analysis.

- `src/evaluation/plots/`  
  Plotting code for the paper-facing figures.

- `src/evaluation/sig_testing/`  
  Significance-testing builders for context-vs-standard comparisons.

- `src/evaluation/translation_audit/`  
  MT1 translation audit pipeline and agreement/error-distribution builders.

---

## Environment and Inputs

This repository assumes a working Python environment with the project dependencies already installed. In practice, the paper-facing scripts should be run with **Python 3.10+**.

You will also need the following external inputs:

1. **Model access**
   - `gpt` runs require an OpenAI API key.
   - translation-audit runs also require OpenAI API access for the judge model.
   - local `llama` and `mistral` runs assume the corresponding model-serving environment used by the repo.

2. **MCQ and math benchmark files**
   - The MCQ/math wrapper scripts expect local benchmark JSON files under `DATASET_DIR`.
   - The expected filenames are:
     - `GlobalMMLU_test.json`
     - `Belebele_test.json`
     - `PIQA_test.json`
     - `MCSQA_test.json`
     - `MGSM_test.json`
     - `MMath_test.json`

3. **MKQA data**
   - The open-ended pipeline expects the MKQA resources from [`apple/ml-mkqa`](https://github.com/apple/ml-mkqa), available locally under `ml-mkqa/` in this repo layout.

4. **MT1 export files**
   - The translation-quality audit scripts expect a directory of MT1 translation exports, passed via `EXPORTS_DIR`.

---

## Open-Ended Evaluation

To reproduce the paper-facing open-ended runs for `aya`, `blend`, `global_piqa`, and `mkqa`:

```bash
script/run_open_ended_final_all.sh
```

This script runs the open-ended cascade pipeline for:

- `llama` = `Llama-3.1-8B-Instruct`
- `mistral` = `Mistral-7B-Instruct-v0.3`
- `gpt` = `GPT-4o-mini`

It writes raw outputs, extracted predictions, and metrics under:

```bash
results/<model>/<dataset>/
```

---

## MCQ and Math Evaluation

To reproduce the paper-facing MCQ and math runs:

```bash
DATASET_DIR=/path/to/dataset script/run_mcq_math_all.sh
```

This runs the evaluation pipeline for the paper’s MCQ and math benchmarks and writes canonical accuracy metrics to:

```bash
results/<model>/<dataset>/metrics/metrics.json
```

The exporter used by this workflow is:

```bash
script/export_pulled_accuracy_metrics.sh
```

---

## Ablations

### Open-ended ablations

```bash
script/run_open_ended_ablation_all.sh
```

### MCQ and math ablations

```bash
DATASET_DIR=/path/to/dataset script/run_mcq_ablation_all.sh
```

These write ablation outputs under:

```bash
results/<model>/<dataset>-ablation/
```

For MCQ and math, variant-specific metrics are written under:

```bash
results/<model>/<dataset>-ablation/<variant>/
```

---

## Figures and Tables

After both the open-ended runs and the MCQ/math runs are available in `results/`, the table and figure builders consume those metrics directly from `results/`.

### Open-ended macro tables

```bash
script/build_open_ended_macro_tables.sh
```

### Ablation average tables

```bash
script/build_ablation_average_tables.sh
```

### Open-ended ablation figure

```bash
script/build_open_ended_ablation_plot.sh
```

### Significance tables

```bash
script/build_significance_tables.sh
```

### Resource-tier violin plots

```bash
script/build_resource_violin_plots.sh
```

These scripts write into:

- `results/open_ended_macro_table/`
- `results/ablation/`
- `results/significance/`
- `results/resource_violin_plots/`

---

## Translation Quality Analysis

### Full LLM judgments

```bash
EXPORTS_DIR=/path/to/mt1_translations script/build_translation_quality_full_llm_judgements.sh
```

### Human-annotated subset judgments

```bash
EXPORTS_DIR=/path/to/mt1_translations script/build_translation_quality_small_llm_judgements.sh
```

### Error-distribution table

```bash
script/build_translation_quality_analysis.sh
```

### Agreement table

```bash
script/build_translation_quality_agreement.sh
```

Outputs are written under:

- `results/translation_quality_analysis/full_llm_judgements/`
- `results/translation_quality_analysis/with_human_judge/`
- `results/translation_quality_analysis/tables/`

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@misc{mazumder2026multilingual,
  title={Multilingual Reasoning Cascades Need More Context},
  author={Mazumder, Arnav and Zhang, Dengjia and Li, Shuyue Stella and Tsvetkov, Yulia and Bafna, Niyati},
  year={2026},
  eprint={2606.27306},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.27306}
}