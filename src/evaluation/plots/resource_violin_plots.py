#!/usr/bin/env python3

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_LEVEL = REPO_ROOT / "resource_level.py"
FEATURES_CSV = REPO_ROOT / "std_cxt_e2e_language_features.csv"

LLAMA_LABEL = "Llama-3.1-8B-Instruct"
MISTRAL_LABEL = "Mistral-7B-Instruct-v0.3"

DATASETS = [
    ("belebele", "Belebele"),
    ("global_piqa", "Global-PIQA-OE"),
    ("mmlu", "MMLU"),
]

RESOURCE_ORDER = ["low_resource", "mid_resource", "high_resource"]
RESOURCE_LABELS = {
    "low_resource": "Low",
    "mid_resource": "Mid",
    "high_resource": "High",
}
RESOURCE_COLORS = {
    "low_resource": "#DC2626",
    "mid_resource": "#D97706",
    "high_resource": "#2563EB",
}

OE_DATASETS = [
    ("aya", "Aya"),
    ("blend", "BLEnD"),
    ("global_piqa", "Global-PIQA-OE"),
    ("mkqa", "MKQA"),
]

AYA_NAMES = {
    "tel": "Telugu",
    "yor": "Yoruba",
    "arb": "Arabic",
    "tur": "Turkish",
    "por": "Portuguese",
    "zho": "Chinese",
}

AYA_SCORE_ALIAS = {"arb": "ara"}

BLEND_NAMES = {
    "CN": "Chinese",
    "ES": "Spanish (ES)",
    "MX": "Spanish (MX)",
    "ID": "Indonesian",
    "KR": "Korean (KR)",
    "KP": "Korean (KP)",
    "GR": "Greek",
    "IR": "Persian",
    "DZ": "Arabic",
    "AZ": "Azerbaijani",
    "JB": "Javanese",
    "AS": "Assamese",
    "NG": "Hausa",
    "ET": "Amharic",
}


def load_resource_level_maps():
    path = RESOURCE_LEVEL
    text = path.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n", ""
    )
    module = ast.parse(text, filename=str(path))
    out = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "_PIQA",
                "_MKQA",
                "_GLOBAL_MMLU",
                "_BELEBELE",
            }:
                out[target.id] = ast.literal_eval(node.value)
    return out["_PIQA"], out["_MKQA"], out["_GLOBAL_MMLU"], out["_BELEBELE"]


PIQA_NAMES, MKQA_NAMES, GLOBAL_MMLU_NAMES, BELEBELE_NAMES = load_resource_level_maps()


def load_resource_score_lookups():
    flores_to_score = {}
    code2_to_score = {}
    code3_to_score = {}
    with FEATURES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flores = str(row["language"]).strip().lower()
            code2 = str(row["language_code"]).strip().lower()
            code3 = str(row["resource_lookup_code"]).strip().lower()
            raw = row["resource_score_log_pages"]
            try:
                score = float(raw)
            except (TypeError, ValueError):
                score = None
            flores_to_score[flores] = score
            if code2 not in code2_to_score or (
                code2_to_score[code2] is None and score is not None
            ):
                code2_to_score[code2] = score
            if code3 not in code3_to_score or (
                code3_to_score[code3] is None and score is not None
            ):
                code3_to_score[code3] = score
    flores_to_score["us"] = flores_to_score.get("eng_latn")
    flores_to_score["gb"] = flores_to_score.get("eng_latn")
    return flores_to_score, code2_to_score, code3_to_score


FLORES_TO_SCORE, CODE2_TO_SCORE, CODE3_TO_SCORE = load_resource_score_lookups()


def resource_score_for_code(code):
    key = code.lower()
    score = FLORES_TO_SCORE.get(key)
    if score is not None:
        return score
    prefix = key.split("_")[0]
    score = FLORES_TO_SCORE.get(prefix)
    if score is not None:
        return score
    return CODE3_TO_SCORE.get(prefix)


def resource_bucket_from_score(score):
    if score is None or score < 12.0:
        return "low_resource"
    if score < 16.0:
        return "mid_resource"
    return "high_resource"


GLOBAL_MMLU_BUCKETS = {
    name: resource_bucket_from_score(CODE2_TO_SCORE.get(code.lower()))
    for code, name in GLOBAL_MMLU_NAMES.items()
}

BELEBELE_BUCKETS = {
    name: resource_bucket_from_score(resource_score_for_code(code))
    for code, name in BELEBELE_NAMES.items()
}


def oe_dataset_codes(model: str, dataset: str):
    config_path = REPO_ROOT / "config" / f"{model}_{dataset}.json"
    config = json.load(config_path.open(encoding="utf-8"))
    return config["dataset"]["langs"]


def oe_dataset_name(dataset: str, code: str):
    if dataset == "aya":
        return AYA_NAMES[code]
    if dataset == "blend":
        return BLEND_NAMES[code]
    if dataset == "global_piqa":
        return PIQA_NAMES[code]
    if dataset == "mkqa":
        return MKQA_NAMES[code]
    raise KeyError(dataset)


def oe_dataset_score_code(dataset: str, code: str):
    if dataset == "aya":
        return AYA_SCORE_ALIAS.get(code, code)
    if dataset == "blend":
        return code.lower()
    return code


def metrics_path(model: str, dataset: str) -> Path:
    candidates = [
        REPO_ROOT / "results" / model / dataset / "metrics" / "metrics.json",
        REPO_ROOT / "results2" / "final" / model / dataset / "metrics" / "metrics.json",
        Path("results/final") / model / dataset / "metrics" / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_accuracy_summary_rows(section_title: str, dataset_key: str, bucket_map):
    path = Path("accuracy_summary_mistral.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_prefix = f"## {section_title}"
    start = next(
        idx for idx, line in enumerate(lines) if line.strip().startswith(header_prefix)
    )
    end = next(
        (idx for idx in range(start + 1, len(lines)) if lines[idx].startswith("## ")),
        len(lines),
    )
    section = lines[start:end]
    rows = []
    in_lang_table = False
    for line in section:
        if line.startswith("| Language |"):
            in_lang_table = True
            continue
        if not in_lang_table:
            continue
        if not line.startswith("|"):
            in_lang_table = False
            continue
        if line.startswith("|----------") or line.startswith("| **Avg**"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        name = parts[0]
        if name not in bucket_map:
            continue
        standard = float(parts[1].replace("%", "").replace("*", ""))
        context = float(parts[3].replace("%", "").replace("*", ""))
        rows.append(
            {
                "dataset": dataset_key,
                "language": name,
                "resource_bucket": bucket_map[name],
                "cxt_minus_standard": context - standard,
            }
        )
    return rows


def read_accuracy_metrics_rows(model: str, dataset: str, bucket_map):
    metrics = json.load(metrics_path(model, dataset).open(encoding="utf-8"))
    std = metrics["slices"]["baseline/standard"]["by_language"]
    ctx = metrics["slices"]["baseline/context"]["by_language"]
    rows = []
    for lang, std_blob in std.items():
        if lang not in ctx or lang not in bucket_map:
            continue
        rows.append(
            {
                "dataset": dataset,
                "language": lang,
                "resource_bucket": bucket_map[lang],
                "cxt_minus_standard": float(ctx[lang]["accuracy"]) - float(std_blob["accuracy"]),
            }
        )
    return rows


def read_oe_actual_rows(model: str, dataset: str):
    metrics = json.load(metrics_path(model, dataset).open(encoding="utf-8"))
    std = metrics["slices"]["baseline/standard"]["by_language"]
    ctx = metrics["slices"]["baseline/context"]["by_language"]
    rows = []
    for code in oe_dataset_codes(model, dataset):
        if code not in std or code not in ctx:
            continue
        bucket = resource_bucket_from_score(
            resource_score_for_code(oe_dataset_score_code(dataset, code))
        )
        rows.append(
            {
                "dataset": dataset,
                "language": code,
                "resource_bucket": bucket,
                "cxt_minus_standard": float(ctx[code]["chrf"]) - float(std[code]["chrf"]),
            }
        )
    return rows


def read_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "language": row["language"],
                        "resource_bucket": row["resource_bucket"],
                        "cxt_minus_standard": float(row["cxt_minus_standard"]),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows


def global_piqa_metrics_path(model: str) -> Path:
    candidates = [
        REPO_ROOT / "results" / model / "global_piqa" / "metrics" / "metrics.json",
        REPO_ROOT / "results2" / "final" / model / "global_piqa" / "metrics" / "metrics.json",
        Path("results/final") / model / "global_piqa" / "metrics" / "metrics.json",
        Path("results/final3") / model / "global_piqa" / "metrics" / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_global_piqa_actual_rows(base: Path, model: str):
    paired_model = "llama" if model == "mistral" else model
    paired_path = base / paired_model / "context_advantage" / "tables" / "context_advantage_paired.csv"
    metrics_path = global_piqa_metrics_path(model)
    resource_by_lang = {}
    with paired_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "global_piqa":
                continue
            if row["metric_name"] != "chrf":
                continue
            resource_by_lang[row["language"]] = row["resource_bucket"]

    metrics = json.load(metrics_path.open(encoding="utf-8"))
    std = metrics["slices"]["baseline/standard"]["by_language"]
    ctx = metrics["slices"]["baseline/context"]["by_language"]
    rows = []
    for lang, bucket in sorted(resource_by_lang.items()):
        if lang not in std or lang not in ctx:
            continue
        rows.append(
            {
                "dataset": "global_piqa",
                "language": lang,
                "resource_bucket": bucket,
                "cxt_minus_standard": float(ctx[lang]["chrf"]) - float(std[lang]["chrf"]),
            }
        )
    return rows


def summarize(rows):
    out = []
    for dataset, _ in DATASETS:
        sub = [r for r in rows if r["dataset"] == dataset]
        for bucket in RESOURCE_ORDER:
            bucket_rows = [r for r in sub if r["resource_bucket"] == bucket]
            values = [r["cxt_minus_standard"] for r in bucket_rows]
            out.append(
                {
                    "dataset": dataset,
                    "resource_bucket": bucket,
                    "mean_cxt_minus_standard": float(np.mean(values)) if values else float("nan"),
                    "n": len(bucket_rows),
                }
            )
    return out


def save_summary_csv(summary_rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "resource_bucket", "mean_cxt_minus_standard", "n"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def add_violin_panel(ax, rows, dataset: str, dataset_label: str, show_ylabel: bool, xlabel: str = "Resource level", ylabel: str = "Context - Standard"):
    sub = [r for r in rows if r["dataset"] == dataset]
    bucket_values = [
        [r["cxt_minus_standard"] for r in sub if r["resource_bucket"] == bucket]
        for bucket in RESOURCE_ORDER
    ]
    positions = np.arange(1, len(RESOURCE_ORDER) + 1)

    nonempty = [(pos, bucket, values) for pos, bucket, values in zip(positions, RESOURCE_ORDER, bucket_values) if values]
    if nonempty:
        violin = ax.violinplot(
            [values for _, _, values in nonempty],
            positions=[pos for pos, _, _ in nonempty],
            widths=0.82,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for body, (_, bucket, _) in zip(violin["bodies"], nonempty):
            body.set_facecolor(RESOURCE_COLORS[bucket])
            body.set_edgecolor(RESOURCE_COLORS[bucket])
            body.set_alpha(0.3)

    for x, values, bucket in zip(positions, bucket_values, RESOURCE_ORDER):
        if not values:
            continue
        ax.scatter(
            [x] * len(values),
            values,
            s=9,
            alpha=0.35,
            color=RESOURCE_COLORS[bucket],
            edgecolors="none",
            zorder=3,
        )
        mean = float(np.mean(values))
        ax.hlines(mean, x - 0.22, x + 0.22, color=RESOURCE_COLORS[bucket], linewidth=2.0, zorder=4)
        ax.text(x, ax.get_ylim()[0], f"n={len(values)}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels([RESOURCE_LABELS[b] for b in RESOURCE_ORDER])
    ax.set_xlabel(xlabel)
    ax.set_title(dataset_label, fontsize=10)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    if show_ylabel:
        ax.set_ylabel(ylabel)


def save_model_plot(rows, model_label: str, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=True)
    y_values = [r["cxt_minus_standard"] for r in rows]
    ymin = min(y_values) if y_values else -0.1
    ymax = max(y_values) if y_values else 0.1
    ypad = 0.08 * max(ymax - ymin, 0.1)
    ymin -= ypad
    ymax += ypad

    for idx, ((dataset, dataset_label), ax) in enumerate(zip(DATASETS, axes)):
        ax.set_ylim(ymin, ymax)
        add_violin_panel(ax, rows, dataset, dataset_label, show_ylabel=(idx == 0))

    fig.suptitle(model_label, fontsize=11, y=1.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_single_dataset_plot(rows, dataset: str, dataset_label: str, output: Path):
    fig, ax = plt.subplots(1, 1, figsize=(2.7, 2.7))
    y_values = [r["cxt_minus_standard"] for r in rows if r["dataset"] == dataset]
    ymin = min(y_values) if y_values else -0.1
    ymax = max(y_values) if y_values else 0.1
    ypad = 0.08 * max(ymax - ymin, 0.1)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    add_violin_panel(
        ax,
        rows,
        dataset,
        dataset_label,
        show_ylabel=True,
        xlabel="",
        ylabel=r"$C_{ctx} - C_{std}$",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_two_model_dataset_plot(
    left_rows,
    right_rows,
    dataset: str,
    dataset_label: str,
    output: Path,
    *,
    left_label: str = LLAMA_LABEL,
    right_label: str = "Qwen",
):
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), sharey=True)
    combined = [r["cxt_minus_standard"] for r in left_rows if r["dataset"] == dataset]
    combined += [r["cxt_minus_standard"] for r in right_rows if r["dataset"] == dataset]
    ymin = min(combined) if combined else -0.1
    ymax = max(combined) if combined else 0.1
    ypad = 0.08 * max(ymax - ymin, 0.1)
    for ax, rows, title, show_ylabel in [
        (axes[0], left_rows, left_label, True),
        (axes[1], right_rows, right_label, False),
    ]:
        ax.set_ylim(ymin - ypad, ymax + ypad)
        add_violin_panel(
            ax,
            rows,
            dataset,
            title,
            show_ylabel=show_ylabel,
            xlabel="",
            ylabel="C$_{ctx}$ - C$_{std}$",
        )
    fig.suptitle(dataset_label, fontsize=10, y=1.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_four_dataset_two_model_plot(dataset_rows, output: Path):
    fig, axes = plt.subplots(4, 2, figsize=(5.2, 9.2), sharex=False, sharey=False)
    for row_idx, (dataset, dataset_label, left_rows, right_rows) in enumerate(dataset_rows):
        combined = [r["cxt_minus_standard"] for r in left_rows if r["dataset"] == dataset]
        combined += [r["cxt_minus_standard"] for r in right_rows if r["dataset"] == dataset]
        ymin = min(combined) if combined else -0.1
        ymax = max(combined) if combined else 0.1
        ypad = 0.08 * max(ymax - ymin, 0.1)
        for col_idx, (rows, title, show_ylabel) in enumerate(
            [(left_rows, LLAMA_LABEL, True), (right_rows, MISTRAL_LABEL, False)]
        ):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(ymin - ypad, ymax + ypad)
            add_violin_panel(
                ax,
                rows,
                dataset,
                title,
                show_ylabel=show_ylabel,
                xlabel="" if row_idx < len(dataset_rows) - 1 else "",
                ylabel="C$_{ctx}$ - C$_{std}$",
            )
        axes[row_idx, 0].text(
            -0.32,
            1.07,
            dataset_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=1.0, w_pad=0.7)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_two_dataset_two_model_plot(dataset_rows, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.8), sharex=False, sharey=False)
    row_titles = []
    for row_idx, (dataset, dataset_label, left_rows, right_rows) in enumerate(dataset_rows):
        combined = [r["cxt_minus_standard"] for r in left_rows if r["dataset"] == dataset]
        combined += [r["cxt_minus_standard"] for r in right_rows if r["dataset"] == dataset]
        ymin = min(combined) if combined else -0.1
        ymax = max(combined) if combined else 0.1
        ypad = 0.08 * max(ymax - ymin, 0.1)
        for col_idx, (rows, title, show_ylabel) in enumerate(
            [(left_rows, LLAMA_LABEL, True), (right_rows, MISTRAL_LABEL, False)]
        ):
            ax = axes[row_idx, col_idx]
            ax.set_ylim(ymin - ypad, ymax + ypad)
            add_violin_panel(
                ax,
                rows,
                dataset,
                title,
                show_ylabel=show_ylabel,
                xlabel="",
                ylabel="C$_{ctx}$ - C$_{std}$",
            )
        row_titles.append((row_idx, dataset_label))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=2.0, w_pad=0.7)
    for row_idx, dataset_label in row_titles:
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        x = (left_pos.x0 + right_pos.x1) / 2.0
        y = max(left_pos.y1, right_pos.y1) + (0.05 if row_idx == 0 else 0.055)
        fig.text(x, y, dataset_label, ha="center", va="bottom", fontsize=10)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "resource_violin_plots",
        help="Directory where the violin plot PDFs will be written.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "llama",
            "qwen",
            "llama_global_piqa",
            "llama_qwen_global_piqa",
            "llama_mistral_global_piqa",
            "llama_mistral_aya",
            "llama_mistral_blend",
            "llama_mistral_global_piqa_oe",
            "llama_mistral_mkqa",
            "llama_mistral_oe",
            "llama_mistral_mmlu",
            "llama_mistral_belebele",
            "llama_mistral_mmlu_belebele",
        ],
        default=[
            "llama_mistral_global_piqa",
            "llama_mistral_mmlu_belebele",
        ],
        help="Subset of violin plots to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path("src/evaluation/regression/final/combined")
    output_dir = args.output_dir
    selected = set(args.plots)

    if "llama" in selected:
        rows = read_rows(base / "llama" / "context_advantage" / "tables" / "context_advantage_paired.csv")
        output = output_dir / "llama_context_minus_standard_by_resource_violin.pdf"
        save_model_plot(rows, "Llama", output)
        print(output)

    if "qwen" in selected:
        rows = read_rows(base / "qwen" / "context_advantage" / "tables" / "context_advantage_paired.csv")
        output = output_dir / "qwen_context_minus_standard_by_resource_violin.pdf"
        save_model_plot(rows, "Qwen", output)
        print(output)

    if {"llama_global_piqa", "llama_qwen_global_piqa", "llama_mistral_global_piqa"} & selected:
        llama_gpqa_rows = read_global_piqa_actual_rows(base, "llama")
        if "llama_global_piqa" in selected:
            out = output_dir / "llama_global_piqa_context_minus_standard_by_resource_violin.pdf"
            save_single_dataset_plot(llama_gpqa_rows, "global_piqa", "Global-PIQA-OE", out)
            print(out)
        if "llama_qwen_global_piqa" in selected:
            qwen_gpqa_rows = read_global_piqa_actual_rows(base, "qwen")
            out = output_dir / "llama_qwen_global_piqa_context_minus_standard_by_resource_violin.pdf"
            save_two_model_dataset_plot(llama_gpqa_rows, qwen_gpqa_rows, "global_piqa", "Global-PIQA-OE", out)
            print(out)
        if "llama_mistral_global_piqa" in selected:
            mistral_gpqa_rows = read_global_piqa_actual_rows(base, "mistral")
            out = output_dir / "llama_mistral_global_piqa_context_minus_standard_by_resource_violin.pdf"
            save_two_model_dataset_plot(
                llama_gpqa_rows,
                mistral_gpqa_rows,
                "global_piqa",
                "Global-PIQA-OE",
                out,
                left_label=LLAMA_LABEL,
                right_label=MISTRAL_LABEL,
            )
            print(out)

    for dataset, dataset_label, plot_key in [
        ("aya", "Aya", "llama_mistral_aya"),
        ("blend", "BLEnD", "llama_mistral_blend"),
        ("global_piqa", "Global-PIQA-OE", "llama_mistral_global_piqa_oe"),
        ("mkqa", "MKQA", "llama_mistral_mkqa"),
    ]:
        if plot_key not in selected:
            continue
        llama_rows = read_oe_actual_rows("llama", dataset)
        mistral_rows = read_oe_actual_rows("mistral", dataset)
        out = output_dir / f"llama_mistral_{dataset}_context_minus_standard_by_resource_violin.pdf"
        save_two_model_dataset_plot(
            llama_rows,
            mistral_rows,
            dataset,
            dataset_label,
            out,
            left_label=LLAMA_LABEL,
            right_label=MISTRAL_LABEL,
        )
        print(out)

    if "llama_mistral_oe" in selected:
        out = output_dir / "llama_mistral_oe_context_minus_standard_by_resource_violin.pdf"
        save_four_dataset_two_model_plot(
            [
                ("aya", "Aya", read_oe_actual_rows("llama", "aya"), read_oe_actual_rows("mistral", "aya")),
                ("blend", "BLEnD", read_oe_actual_rows("llama", "blend"), read_oe_actual_rows("mistral", "blend")),
                ("global_piqa", "Global-PIQA-OE", read_oe_actual_rows("llama", "global_piqa"), read_oe_actual_rows("mistral", "global_piqa")),
                ("mkqa", "MKQA", read_oe_actual_rows("llama", "mkqa"), read_oe_actual_rows("mistral", "mkqa")),
            ],
            out,
        )
        print(out)

    if {"llama_mistral_mmlu", "llama_mistral_belebele", "llama_mistral_mmlu_belebele"} & selected:
        llama_mmlu_rows = read_accuracy_metrics_rows("llama", "mmlu", GLOBAL_MMLU_BUCKETS)
        if metrics_path("mistral", "mmlu").exists():
            mistral_mmlu_rows = read_accuracy_metrics_rows("mistral", "mmlu", GLOBAL_MMLU_BUCKETS)
        else:
            mistral_mmlu_rows = read_accuracy_summary_rows("Global MMLU", "mmlu", GLOBAL_MMLU_BUCKETS)
        llama_belebele_rows = read_accuracy_metrics_rows("llama", "belebele", BELEBELE_BUCKETS)
        if metrics_path("mistral", "belebele").exists():
            mistral_belebele_rows = read_accuracy_metrics_rows("mistral", "belebele", BELEBELE_BUCKETS)
        else:
            mistral_belebele_rows = read_accuracy_summary_rows("Belebele", "belebele", BELEBELE_BUCKETS)

        if "llama_mistral_mmlu" in selected:
            out = output_dir / "llama_mistral_mmlu_context_minus_standard_by_resource_violin.pdf"
            save_two_model_dataset_plot(
                llama_mmlu_rows,
                mistral_mmlu_rows,
                "mmlu",
                "Global MMLU",
                out,
                left_label=LLAMA_LABEL,
                right_label=MISTRAL_LABEL,
            )
            print(out)

        if "llama_mistral_belebele" in selected:
            out = output_dir / "llama_mistral_belebele_context_minus_standard_by_resource_violin.pdf"
            save_two_model_dataset_plot(
                llama_belebele_rows,
                mistral_belebele_rows,
                "belebele",
                "Belebele",
                out,
                left_label=LLAMA_LABEL,
                right_label=MISTRAL_LABEL,
            )
            print(out)

        if "llama_mistral_mmlu_belebele" in selected:
            out = output_dir / "llama_mistral_mmlu_belebele_context_minus_standard_by_resource_violin.pdf"
            save_two_dataset_two_model_plot(
                [
                    ("mmlu", "Global MMLU", llama_mmlu_rows, mistral_mmlu_rows),
                    ("belebele", "Belebele", llama_belebele_rows, mistral_belebele_rows),
                ],
                out,
            )
            print(out)


if __name__ == "__main__":
    main()
