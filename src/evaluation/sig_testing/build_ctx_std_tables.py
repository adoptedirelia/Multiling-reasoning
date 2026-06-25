#!/usr/bin/env python3

import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple

from .final_ctx_std_tests import DATASET_ORDER, compute_ctx_std_tests_for_model


MODELS: List[Tuple[str, str]] = [
    ("llama", "Llama-3.1-8B-Instruct"),
    ("mistral", "Mistral-7B-Instruct-v0.3"),
    ("gpt", "GPT-4o-mini"),
]

DATASET_GROUPS = {
    "aya": "open_ended",
    "blend": "open_ended",
    "global_piqa": "open_ended",
    "mkqa": "open_ended",
    "mmlu": "multiple_choice",
    "belebele": "multiple_choice",
    "global-piqa-mc": "multiple_choice",
    "mcsqa": "multiple_choice",
    "mgsm": "math",
    "mmath": "math",
}

CURRENT_MODEL_DIR = {
    "llama": "llama",
    "mistral": "mistral",
    "gpt": "gpt",
}

LEGACY_MODEL_DIR = {
    "llama": "llama",
    "mistral": "mistral",
    "gpt": "gpt-30-samples",
}


def round_half_up(value: float, ndigits: int) -> float:
    quantum = "1." + ("0" * ndigits)
    return float(Decimal(str(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


def subtract_rounded(a: float, b: float, ndigits: int) -> float:
    quantum = Decimal("1." + ("0" * ndigits))
    a_dec = Decimal(("{0:." + str(ndigits) + "f}").format(a))
    b_dec = Decimal(("{0:." + str(ndigits) + "f}").format(b))
    return float((a_dec - b_dec).quantize(quantum, rounding=ROUND_HALF_UP))


def resolve_metrics_path(repo_root: Path, model: str, dataset_key: str) -> Path:
    current = repo_root / "results" / CURRENT_MODEL_DIR[model] / dataset_key / "metrics" / "metrics.json"
    if current.exists():
        return current
    legacy = repo_root / "results2" / "final" / LEGACY_MODEL_DIR[model] / dataset_key / "metrics" / "metrics.json"
    return legacy


def load_macro_average(metrics_path: Path, slice_key: str) -> float:
    blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_language = blob["slices"][slice_key]["by_language"]
    sample = next(iter(by_language.values()))
    metric_name = None
    for key in ("chrf", "accuracy", "f1"):
        if key in sample:
            metric_name = key
            break
    if metric_name is None:
        raise ValueError("Could not determine metric field")
    values = [float(stats[metric_name]) for stats in by_language.values()]
    return sum(values) / len(values)


def _to_row_map(rows) -> Dict[str, object]:
    return {row.dataset_key: row for row in rows}


def _direction(mean_delta: float) -> str:
    if math.isnan(mean_delta):
        return "neutral"
    if mean_delta > 0:
        return "positive"
    if mean_delta < 0:
        return "negative"
    return "neutral"


def build_tables(repo_root: Path) -> Dict[str, List[Dict]]:
    per_model = {}
    for model_key, _model_label in MODELS:
        per_model[model_key] = _to_row_map(
            compute_ctx_std_tests_for_model(
                repo_root,
                model_key,
                metrics_path_resolver=resolve_metrics_path,
            )
        )

    difference_rows = []
    win_rate_rows = []
    wilcoxon_rows = []
    binomial_rows = []

    for dataset_key, dataset_label in DATASET_ORDER:
        available = [per_model[model_key].get(dataset_key) for model_key, _ in MODELS]
        available = [row for row in available if row is not None]
        if not available:
            continue

        base_row = available[0]
        group = DATASET_GROUPS.get(dataset_key, "other")

        diff_row = {
            "dataset_key": dataset_key,
            "dataset": dataset_label,
            "task_group": group,
            "metric": base_row.metric,
        }
        win_row = {
            "dataset_key": dataset_key,
            "dataset": dataset_label,
            "task_group": group,
            "metric": base_row.metric,
            "n_languages": int(base_row.n_languages),
        }
        wil_row = {
            "dataset_key": dataset_key,
            "dataset": dataset_label,
            "task_group": group,
            "metric": base_row.metric,
        }
        bin_row = {
            "dataset_key": dataset_key,
            "dataset": dataset_label,
            "task_group": group,
            "metric": base_row.metric,
            "n_languages": int(base_row.n_languages),
        }

        for model_key, model_label in MODELS:
            row = per_model[model_key].get(dataset_key)
            if row is None:
                diff_row[model_key] = None
                win_row[model_key] = None
                wil_row[model_key] = None
                bin_row[model_key] = None
                continue

            metrics_path = resolve_metrics_path(repo_root, model_key, dataset_key)
            std_avg = round_half_up(load_macro_average(metrics_path, "baseline/standard"), 2)
            ctx_avg = round_half_up(load_macro_average(metrics_path, "baseline/context"), 2)
            diff_row[model_key] = subtract_rounded(ctx_avg, std_avg, 2)
            if math.isnan(row.win_rate):
                win_row[model_key] = None
            else:
                win_row[model_key] = round_half_up(100.0 * row.win_rate, 1)

            wil_row[model_key] = {
                "model_label": model_label,
                "q_value": None if math.isnan(row.wilcoxon_q_bh) else row.wilcoxon_q_bh,
                "direction": _direction(row.mean_delta),
                "significant": (not math.isnan(row.wilcoxon_q_bh)) and row.wilcoxon_q_bh < 0.05,
            }
            bin_row[model_key] = {
                "model_label": model_label,
                "q_value": None if math.isnan(row.binomial_q_bh) else row.binomial_q_bh,
                "direction": _direction(row.mean_delta),
                "significant": (not math.isnan(row.binomial_q_bh)) and row.binomial_q_bh < 0.05,
            }

        difference_rows.append(diff_row)
        win_rate_rows.append(win_row)
        wilcoxon_rows.append(wil_row)
        binomial_rows.append(bin_row)

    return {
        "ctx_vs_std_mean_delta_table": difference_rows,
        "ctx_vs_std_win_rate_table": win_rate_rows,
        "ctx_vs_std_wilcoxon_q_table": wilcoxon_rows,
        "ctx_vs_std_binomial_q_table": binomial_rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ctx-vs-std summary/significance tables.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "results" / "significance"
    tables = build_tables(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_json(out_dir / (name + ".json"), rows)
        print(out_dir / (name + ".json"))


if __name__ == "__main__":
    main()
