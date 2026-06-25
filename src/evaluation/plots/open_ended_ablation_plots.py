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
PLOTS_DIR = REPO_ROOT / "results" / "ablation" / "plots"
RESOURCE_LEVEL = REPO_ROOT / "resource_level.py"
FEATURES_CSV = REPO_ROOT / "std_cxt_e2e_language_features.csv"

DATASETS = [
    ("aya", "Aya"),
    ("blend", "BLEnD"),
    ("global_piqa", "Global-PIQA-OE"),
    ("mkqa", "MKQA"),
]

ALL_DATASETS = [
    ("aya", "Aya", "oe"),
    ("blend", "BLEnD", "oe"),
    ("global_piqa", "Global-PIQA-OE", "oe"),
    ("mkqa", "MKQA", "oe"),
    ("mmlu", "Global MMLU", "report"),
    ("belebele", "Belebele", "report"),
    ("global-piqa-mc", "Global-PIQA", "report"),
    ("mcsqa", "MCSQA", "report"),
    ("mgsm", "MGSM", "report"),
    ("mmath", "MMath", "report"),
]

CHRf_DATASETS = [item for item in ALL_DATASETS if item[2] == "oe"]
ACCURACY_DATASETS = [item for item in ALL_DATASETS if item[2] == "report"]

METHODS = [
    ("context", r"$C_{ctx}$", "#1d4ed8"),
    ("answer_plus_source_question", r"$q_t + a_e$", "#059669"),
    ("answer_plus_english_question", r"$q_e + a_e$", "#d97706"),
    ("answer_plus_reasoning", r"$r_e + a_e$", "#dc2626"),
]

RESOURCE_ORDER = ["low", "mid", "high"]
RESOURCE_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}

AYA_NAMES = {
    "arb": "Arabic",
    "por": "Portuguese",
    "tel": "Telugu",
    "tur": "Turkish",
    "yor": "Yoruba",
    "zho": "Chinese",
}

BLEND_NAMES = {
    "AZ": "Azerbaijani",
    "CN": "Chinese",
    "DZ": "Arabic",
    "ES": "Spanish (ES)",
    "ET": "Amharic",
    "GR": "Greek",
    "ID": "Indonesian",
    "IR": "Persian",
    "JB": "Javanese",
    "KP": "Korean (KP)",
    "KR": "Korean (KR)",
    "MX": "Spanish (MX)",
    "NG": "Hausa",
    "AS": "Assamese",
}

MKQA_NAMES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "km": "Khmer",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh_cn": "Chinese (Simplified)",
    "zh_hk": "Chinese (HK)",
    "zh_tw": "Chinese (Traditional)",
}


def load_resource_maps():
    text = RESOURCE_LEVEL.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n", ""
    )
    module = ast.parse(text, filename=str(RESOURCE_LEVEL))
    out = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PIQA":
                out[target.id] = ast.literal_eval(node.value)
    return out["_PIQA"]


PIQA_NAMES = load_resource_maps()
ACCURACY_DATASET_KEYS = {
    "Global MMLU": "mmlu",
    "Belebele": "belebele",
    "Global-PIQA": "global-piqa-mc",
    "MCSQA": "mcsqa",
    "MGSM": "mgsm",
    "MMath": "mmath",
}


def load_score_lookup():
    by_lang = {}
    by_code2 = {}
    by_code3 = {}
    with FEATURES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["resource_score_log_pages"])
            except Exception:
                continue
            by_lang.setdefault(row["language"], score)
            by_code2.setdefault(row["language_code"], score)
            by_code3.setdefault(row["resource_lookup_code"], score)
    return by_lang, by_code2, by_code3


BY_LANG, BY_CODE2, BY_CODE3 = load_score_lookup()


def score_for(code):
    code = code.lower()
    if code in BY_LANG:
        return BY_LANG[code]
    if code in BY_CODE2:
        return BY_CODE2[code]
    if code in BY_CODE3:
        return BY_CODE3[code]
    if "_" in code:
        prefix = code.split("_")[0]
        if prefix in BY_LANG:
            return BY_LANG[prefix]
        if prefix in BY_CODE2:
            return BY_CODE2[prefix]
        if prefix in BY_CODE3:
            return BY_CODE3[prefix]
    return None


def resource_bucket(code):
    score = score_for(code)
    if score is None or score < 12.0:
        return "low"
    if score < 16.0:
        return "mid"
    return "high"


def dataset_name_map(dataset):
    if dataset == "aya":
        return AYA_NAMES
    if dataset == "blend":
        return BLEND_NAMES
    if dataset == "global_piqa":
        return PIQA_NAMES
    if dataset == "mkqa":
        return MKQA_NAMES
    raise KeyError(dataset)


def metrics_path(model, dataset, variant=None):
    if variant is None:
        return REPO_ROOT / "results" / model / dataset / "metrics" / "metrics.json"
    return REPO_ROOT / "results" / model / (dataset + "-ablation") / variant / "metrics" / "metrics.json"


def load_metric_map(path, slice_name):
    obj = json.loads(path.read_text())
    by_language = obj["slices"][slice_name]["by_language"]
    sample = next(iter(by_language.values()))
    metric_name = "chrf" if "chrf" in sample else "accuracy"
    return {
        key: float(value[metric_name])
        for key, value in by_language.items()
    }


def load_slice(path, slice_name):
    return load_metric_map(path, slice_name)


def load_slice_blob(path, slice_name):
    obj = json.loads(path.read_text())
    return obj["slices"][slice_name]


def load_accuracy_dataset_overall_from_metrics(model, dataset_key):
    values = {}
    std_slice = load_slice_blob(metrics_path(model, dataset_key), "baseline/standard")
    ctx_slice = load_slice_blob(metrics_path(model, dataset_key), "baseline/context")
    values["standard"] = float(std_slice["overall"])
    values["context"] = float(ctx_slice["overall"])
    for method, _, _ in METHODS:
        if method == "context":
            continue
        values[method] = float(
            load_slice_blob(
                metrics_path(model, dataset_key, method),
                f"baseline/{method}",
            )["overall"]
        )
    return values


def load_accuracy_dataset_resources_from_metrics(model, dataset_key):
    values = {bucket: {} for bucket in RESOURCE_ORDER}
    slices = {
        "standard": load_slice_blob(metrics_path(model, dataset_key), "baseline/standard"),
        "context": load_slice_blob(metrics_path(model, dataset_key), "baseline/context"),
    }
    for method, _, _ in METHODS:
        if method == "context":
            continue
        slices[method] = load_slice_blob(
            metrics_path(model, dataset_key, method),
            f"baseline/{method}",
        )
    for bucket in RESOURCE_ORDER:
        for method_name, slice_blob in slices.items():
            values[bucket][method_name] = float(
                slice_blob.get("by_resource", {}).get(bucket, {}).get("overall", np.nan)
            )
    return values


def load_dataset_rows(model, dataset):
    std = load_slice(metrics_path(model, dataset), "baseline/standard")
    ctx = load_slice(metrics_path(model, dataset), "baseline/context")
    srcq = load_slice(
        metrics_path(model, dataset, "answer_plus_source_question"),
        "baseline/answer_plus_source_question",
    )
    enq = load_slice(
        metrics_path(model, dataset, "answer_plus_english_question"),
        "baseline/answer_plus_english_question",
    )
    reas = load_slice(
        metrics_path(model, dataset, "answer_plus_reasoning"),
        "baseline/answer_plus_reasoning",
    )
    name_map = dataset_name_map(dataset)
    rows = []
    for code, label in name_map.items():
        rows.append(
            {
                "code": code,
                "label": label,
                "bucket": resource_bucket(code),
                "standard": std[code],
                "context": ctx[code],
                "answer_plus_source_question": srcq[code],
                "answer_plus_english_question": enq[code],
                "answer_plus_reasoning": reas[code],
            }
        )
    return rows


def delta_means(model):
    out = {}
    for dataset, _ in DATASETS:
        rows = load_dataset_rows(model, dataset)
        by_method = {}
        for method, _, _ in METHODS:
            by_method[method] = np.mean([r[method] - r["standard"] for r in rows])
        out[dataset] = by_method
    return out


def delta_means_all(model):
    out = {}
    for dataset, label, source in ALL_DATASETS:
        if source == "oe":
            rows = load_dataset_rows(model, dataset)
            out[dataset] = {}
            for method, _, _ in METHODS:
                out[dataset][method] = np.mean([r[method] - r["standard"] for r in rows])
        else:
            dataset_key = ACCURACY_DATASET_KEYS[label]
            row = load_accuracy_dataset_overall_from_metrics(model, dataset_key)
            out[dataset] = {}
            for method, _, _ in METHODS:
                out[dataset][method] = row[method] - row["standard"]
    return out


def percent_means_all(model):
    out = {}
    for dataset, label, source in ALL_DATASETS:
        out[dataset] = {}
        if source == "oe":
            rows = load_dataset_rows(model, dataset)
            for method, _, _ in METHODS:
                vals = []
                for row in rows:
                    std = row["standard"]
                    vals.append(0.0 if std == 0 else 100.0 * (row[method] - std) / std)
                out[dataset][method] = np.mean(vals)
        else:
            dataset_key = ACCURACY_DATASET_KEYS[label]
            row = load_accuracy_dataset_overall_from_metrics(model, dataset_key)
            std = row["standard"]
            for method, _, _ in METHODS:
                out[dataset][method] = 0.0 if std == 0 else 100.0 * (row[method] - std) / std
    return out


def resource_means(model):
    out = {}
    for dataset, _ in DATASETS:
        rows = load_dataset_rows(model, dataset)
        out[dataset] = {}
        for bucket in RESOURCE_ORDER:
            bucket_rows = [r for r in rows if r["bucket"] == bucket]
            out[dataset][bucket] = {}
            for method, _, _ in METHODS:
                if bucket_rows:
                    out[dataset][bucket][method] = np.mean([r[method] for r in bucket_rows])
                else:
                    out[dataset][bucket][method] = np.nan
    return out


def resource_means_all(model):
    out = {}
    for dataset, label, source in ALL_DATASETS:
        if source == "oe":
            rows = load_dataset_rows(model, dataset)
            out[dataset] = {}
            for bucket in RESOURCE_ORDER:
                bucket_rows = [r for r in rows if r["bucket"] == bucket]
                out[dataset][bucket] = {}
                for method, _, _ in METHODS:
                    if bucket_rows:
                        out[dataset][bucket][method] = np.mean([r[method] for r in bucket_rows])
                    else:
                        out[dataset][bucket][method] = np.nan
        else:
            dataset_key = ACCURACY_DATASET_KEYS[label]
            out[dataset] = load_accuracy_dataset_resources_from_metrics(model, dataset_key)
    return out


def save_delta_plot(output):
    model_panels = [
        ("llama", "Llama-3.1-8B-Instruct"),
        ("mistral", "Mistral-7B-Instruct-v0.3"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    width = 0.18
    x = np.arange(len(DATASETS))

    for ax, (model, title) in zip(axes, model_panels):
        means = delta_means(model)
        for idx, (method, label, color) in enumerate(METHODS):
            vals = [means[dataset][method] for dataset, _ in DATASETS]
            ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in DATASETS], rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel(r"$C - C_{std}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_delta_plot_single_model(model, model_label, output):
    means = delta_means(model)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.0))
    width = 0.18
    x = np.arange(len(DATASETS))

    for idx, (method, label, color) in enumerate(METHODS):
        vals = [means[dataset][method] for dataset, _ in DATASETS]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS], rotation=15, ha="right")
    ax.set_title(model_label, fontsize=10)
    ax.set_ylabel(r"$C - C_{std}$")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_delta_plot_all(output):
    model_panels = [
        ("llama", "Llama-3.1-8B-Instruct"),
        ("mistral", "Mistral-7B-Instruct-v0.3"),
    ]
    metric_panels = [
        ("chrF", CHRf_DATASETS),
        ("Accuracy", ACCURACY_DATASETS),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.6), sharey="col")
    width = 0.18

    for row_idx, (model, title) in enumerate(model_panels):
        means = delta_means_all(model)
        for col_idx, (metric_title, dataset_group) in enumerate(metric_panels):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(dataset_group))
            for idx, (method, label, color) in enumerate(METHODS):
                vals = [means[dataset][method] for dataset, _, _ in dataset_group]
                ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
            ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label for _, label, _ in dataset_group], rotation=22, ha="right"
            )
            if row_idx == 0:
                ax.set_title(metric_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\n" + r"$C - C_{std}$")
            else:
                ax.set_ylabel(r"$C - C_{std}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4, w_pad=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_percent_plot_all(output):
    model_panels = [
        ("llama", "Llama-3.1-8B-Instruct"),
        ("mistral", "Mistral-7B-Instruct-v0.3"),
    ]
    metric_panels = [
        ("chrF", CHRf_DATASETS),
        ("Accuracy", ACCURACY_DATASETS),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.6), sharey="col")
    width = 0.18

    for row_idx, (model, title) in enumerate(model_panels):
        means = percent_means_all(model)
        for col_idx, (metric_title, dataset_group) in enumerate(metric_panels):
            ax = axes[row_idx, col_idx]
            x = np.arange(len(dataset_group))
            for idx, (method, label, color) in enumerate(METHODS):
                vals = [means[dataset][method] for dataset, _, _ in dataset_group]
                ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
            ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [label for _, label, _ in dataset_group], rotation=22, ha="right"
            )
            if row_idx == 0:
                ax.set_title(metric_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\n% increase over " + r"$C_{std}$")
            else:
                ax.set_ylabel(r"% increase over $C_{std}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4, w_pad=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_resource_plot(model, model_label, output):
    means = resource_means(model)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    axes = axes.flatten()
    width = 0.18
    x = np.arange(len(RESOURCE_ORDER))

    for ax, (dataset, dataset_label) in zip(axes, DATASETS):
        for idx, (method, label, color) in enumerate(METHODS):
            vals = [means[dataset][bucket][method] for bucket in RESOURCE_ORDER]
            ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([RESOURCE_LABELS[b] for b in RESOURCE_ORDER])
        ax.set_title(dataset_label, fontsize=10)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel(r"$C$")
    axes[2].set_ylabel(r"$C$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(model_label, fontsize=11, y=1.02)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.1, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_resource_plot_all(model, model_label, output):
    means = resource_means_all(model)
    fig, axes = plt.subplots(5, 2, figsize=(8.4, 10.8), sharey=False)
    axes = axes.flatten()
    width = 0.18
    x = np.arange(len(RESOURCE_ORDER))

    for ax, (dataset, dataset_label, _) in zip(axes, ALL_DATASETS):
        for idx, (method, label, color) in enumerate(METHODS):
            vals = [means[dataset][bucket][method] for bucket in RESOURCE_ORDER]
            ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([RESOURCE_LABELS[b] for b in RESOURCE_ORDER])
        ax.set_title(dataset_label, fontsize=10)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    for ax in axes[::2]:
        ax.set_ylabel(r"$C$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(model_label, fontsize=11, y=1.01)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.0, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory where the ablation plots will be written.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=[
            "llama_mistral_oe_delta",
            "llama_oe_delta",
            "mistral_oe_delta",
            "llama_oe_resource",
            "mistral_oe_resource",
            "llama_mistral_all_delta",
            "llama_mistral_all_percent",
            "llama_all_resource",
            "mistral_all_resource",
        ],
        default=["llama_mistral_oe_delta"],
        help="Subset of ablation plots to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    selected = set(args.plots)
    if "llama_mistral_oe_delta" in selected:
        out = output_dir / "llama_mistral_oe_ablation_delta_over_standard.pdf"
        save_delta_plot(out)
        print(out)
    if "llama_oe_delta" in selected:
        out = output_dir / "llama_oe_ablation_delta_over_standard.pdf"
        save_delta_plot_single_model("llama", "Llama-3.1-8B-Instruct", out)
        print(out)
    if "mistral_oe_delta" in selected:
        out = output_dir / "mistral_oe_ablation_delta_over_standard.pdf"
        save_delta_plot_single_model("mistral", "Mistral-7B-Instruct-v0.3", out)
        print(out)
    if "llama_oe_resource" in selected:
        out = output_dir / "llama_oe_ablation_by_resource_grouped_bars.pdf"
        save_resource_plot("llama", "Llama-3.1-8B-Instruct", out)
        print(out)
    if "mistral_oe_resource" in selected:
        out = output_dir / "mistral_oe_ablation_by_resource_grouped_bars.pdf"
        save_resource_plot("mistral", "Mistral-7B-Instruct-v0.3", out)
        print(out)
    if "llama_mistral_all_delta" in selected:
        out = output_dir / "llama_mistral_all_ablation_delta_over_standard.pdf"
        save_delta_plot_all(out)
        print(out)
    if "llama_mistral_all_percent" in selected:
        out = output_dir / "llama_mistral_all_ablation_percent_over_standard.pdf"
        save_percent_plot_all(out)
        print(out)
    if "llama_all_resource" in selected:
        out = output_dir / "llama_all_ablation_by_resource_grouped_bars.pdf"
        save_resource_plot_all("llama", "Llama-3.1-8B-Instruct", out)
        print(out)
    if "mistral_all_resource" in selected:
        out = output_dir / "mistral_all_ablation_by_resource_grouped_bars.pdf"
        save_resource_plot_all("mistral", "Mistral-7B-Instruct-v0.3", out)
        print(out)


if __name__ == "__main__":
    main()
