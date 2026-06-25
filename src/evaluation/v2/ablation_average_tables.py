#!/usr/bin/env python3

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path("/gscratch/stf/arnav/mt-llm-mt/Multiling-reasoning")
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "ablation" / "tables"
REPORT_FILES = {
    "llama": REPO_ROOT / "ablation_report_llama.md",
    "mistral": REPO_ROOT / "ablation_report_mistral.md",
}

OPEN_ENDED_DATASETS = [
    ("aya", "Aya", "Open-ended generation (chrF)"),
    ("blend", "BLEnD", "Open-ended generation (chrF)"),
    ("global_piqa", "Global-PIQA-OE", "Open-ended generation (chrF)"),
    ("mkqa", "MKQA", "Open-ended generation (chrF)"),
]

REPORT_DATASETS = [
    ("Global MMLU", "Multiple-choice (Acc.)"),
    ("Belebele", "Multiple-choice (Acc.)"),
    ("Global-PIQA", "Multiple-choice (Acc.)"),
    ("MCSQA", "Multiple-choice (Acc.)"),
    ("MGSM", "Math (Acc.)"),
    ("MMath", "Math (Acc.)"),
]

TASK_GROUP_ORDER = [
    "Open-ended generation (chrF)",
    "Multiple-choice (Acc.)",
    "Math (Acc.)",
]

MODEL_LABELS = {
    "llama": "Llama-3.1-8B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
}

METHOD_KEYS = [
    ("standard", "\\cstd"),
    ("context", "\\cctx"),
    ("answer_plus_source_question", "$q_t+a_e$"),
    ("answer_plus_english_question", "$q_e+a_e$"),
    ("answer_plus_reasoning", "$r_e+a_e$"),
]

OE_SLICE_KEYS = {
    "standard": "baseline/standard",
    "context": "baseline/context",
    "answer_plus_source_question": "baseline/answer_plus_source_question",
    "answer_plus_english_question": "baseline/answer_plus_english_question",
    "answer_plus_reasoning": "baseline/answer_plus_reasoning",
}


def round_half_up(value: float, ndigits: int = 2) -> float:
    quantum = "1." + ("0" * ndigits)
    return float(Decimal(str(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def infer_metric_name(by_language: Dict[str, Dict[str, float]]) -> str:
    sample = next(iter(by_language.values()))
    for key in ("chrf", "accuracy", "f1"):
        if key in sample:
            return key
    raise ValueError("Could not infer metric name.")


def load_oe_metric(path: Path, slice_name: str) -> float:
    blob = json.loads(path.read_text(encoding="utf-8"))
    by_language = blob["slices"][slice_name]["by_language"]
    metric_name = infer_metric_name(by_language)
    values = [float(stats[metric_name]) for stats in by_language.values()]
    return round_half_up(mean(values))


def load_oe_rows(model: str) -> List[Dict[str, object]]:
    rows = []
    for dataset_key, dataset_label, task_group in OPEN_ENDED_DATASETS:
        base_metrics = RESULTS_ROOT / model / dataset_key / "metrics" / "metrics.json"
        row = {
            "dataset": dataset_label,
            "task_group": task_group,
            "values": {
                "standard": load_oe_metric(base_metrics, OE_SLICE_KEYS["standard"]),
                "context": load_oe_metric(base_metrics, OE_SLICE_KEYS["context"]),
                "answer_plus_source_question": load_oe_metric(
                    RESULTS_ROOT
                    / model
                    / f"{dataset_key}-ablation"
                    / "answer_plus_source_question"
                    / "metrics"
                    / "metrics.json",
                    OE_SLICE_KEYS["answer_plus_source_question"],
                ),
                "answer_plus_english_question": load_oe_metric(
                    RESULTS_ROOT
                    / model
                    / f"{dataset_key}-ablation"
                    / "answer_plus_english_question"
                    / "metrics"
                    / "metrics.json",
                    OE_SLICE_KEYS["answer_plus_english_question"],
                ),
                "answer_plus_reasoning": load_oe_metric(
                    RESULTS_ROOT
                    / model
                    / f"{dataset_key}-ablation"
                    / "answer_plus_reasoning"
                    / "metrics"
                    / "metrics.json",
                    OE_SLICE_KEYS["answer_plus_reasoning"],
                ),
            },
        }
        rows.append(row)
    return rows


def parse_report_overall(model: str) -> Dict[str, Dict[str, float]]:
    lines = REPORT_FILES[model].read_text(encoding="utf-8").splitlines()
    in_overall = False
    overall = {}
    for line in lines:
        stripped = line.strip()
        if stripped == "## Overall Accuracy":
            in_overall = True
            continue
        if in_overall and stripped.startswith("## "):
            break
        if not in_overall:
            continue
        if stripped.startswith("|") and "Dataset" not in stripped and "---" not in stripped:
            parts = [p.strip().replace("**", "") for p in stripped.strip("|").split("|")]
            if len(parts) != 6:
                continue
            overall[parts[0]] = {
                "standard": float(parts[1].rstrip("%")),
                "context": float(parts[2].rstrip("%")),
                "answer_plus_source_question": float(parts[3].rstrip("%")),
                "answer_plus_english_question": float(parts[4].rstrip("%")),
                "answer_plus_reasoning": float(parts[5].rstrip("%")),
            }
    return overall


def load_report_rows(model: str) -> List[Dict[str, object]]:
    overall = parse_report_overall(model)
    rows = []
    for dataset_label, task_group in REPORT_DATASETS:
        rows.append(
            {
                "dataset": dataset_label,
                "task_group": task_group,
                "values": overall[dataset_label],
            }
        )
    return rows


def build_model_rows(model: str) -> List[Dict[str, object]]:
    return load_oe_rows(model) + load_report_rows(model)


def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def latex_value(value: float, best: float) -> str:
    text = fmt_num(value)
    return f"\\textbf{{{text}}}" if value == best else text


def write_json(model: str, rows: List[Dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{model}_ablation_averages.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_path


def write_latex(model: str, rows: List[Dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{model}_ablation_averages.tex"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & \\cstd & \\cctx & $q_t+a_e$ & $q_e+a_e$ & $r_e+a_e$ \\\\",
        "\\midrule",
    ]
    for task_group in TASK_GROUP_ORDER:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{task_group}}}}} \\\\")
        for row in rows:
            if row["task_group"] != task_group:
                continue
            values = row["values"]
            ordered_values = [values[key] for key, _label in METHOD_KEYS]
            best = max(ordered_values)
            rendered = [latex_value(values[key], best) for key, _label in METHOD_KEYS]
            lines.append(
                f'{row["dataset"]} & {rendered[0]} & {rendered[1]} & {rendered[2]} & {rendered[3]} & {rendered[4]} \\\\'
            )
        if task_group != TASK_GROUP_ORDER[-1]:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{MODEL_LABELS[model]} overall accuracy and chrF scores with varying context provided to \\mttwo on all datasets}}",
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    for model in ("llama", "mistral"):
        rows = build_model_rows(model)
        write_json(model, rows)
        write_latex(model, rows)


if __name__ == "__main__":
    main()
