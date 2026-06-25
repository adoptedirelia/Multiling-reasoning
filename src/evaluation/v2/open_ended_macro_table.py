#!/usr/bin/env python3

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATASETS = [
    ("aya", "Aya", "Open-ended cultural"),
    ("blend", "BLEnD", "Open-ended cultural"),
    ("global_piqa", "Global-PIQA-OE", "Open-ended cultural"),
    ("mkqa", "MKQA", "Open-ended factual"),
]

GAP_CLOSED_ORDER = ["Aya", "Global-PIQA-OE", "BLEnD", "MKQA"]

MODELS = [
    ("llama", "Llama-3.1-8B-Instruct"),
    ("mistral", "Mistral-7B-Instruct-v0.3"),
    ("gpt", "GPT-4o-mini"),
]

SLICES = [
    ("baseline/standard", "\\cstd"),
    ("baseline/direct", "\\etoe"),
    ("baseline/context", "\\cctx"),
]


def round_half_up(value: float, ndigits: int = 2) -> float:
    quantum = "1." + ("0" * ndigits)
    return float(Decimal(str(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))


def infer_metric_name(by_language: dict) -> str:
    sample = next(iter(by_language.values()))
    for key in ("chrf", "accuracy", "f1"):
        if key in sample:
            return key
    raise ValueError("Could not infer metric name from by_language row")


def load_macro_scores(metrics_path: Path) -> Tuple[str, Dict[str, float]]:
    blob = json.loads(metrics_path.read_text(encoding="utf-8"))
    scores = {}  # type: Dict[str, float]
    metric_name = None  # type: Optional[str]
    for slice_key, _label in SLICES:
        by_language = blob["slices"][slice_key]["by_language"]
        if metric_name is None:
            metric_name = infer_metric_name(by_language)
        values = [float(stats[metric_name]) for stats in by_language.values()]
        scores[slice_key] = sum(values) / len(values)
    assert metric_name is not None
    return metric_name, scores


def build_rows(results_root: Path) -> List[Dict]:
    rows = []
    for dataset_key, dataset_label, task_type in DATASETS:
        row = {
            "dataset_key": dataset_key,
            "dataset": dataset_label,
            "task_type": task_type,
            "metric": None,
            "models": {},
        }
        for model_key, model_label in MODELS:
            metrics_path = results_root / model_key / dataset_key / "metrics" / "metrics.json"
            metric_name, scores = load_macro_scores(metrics_path)
            row["metric"] = metric_name
            row["models"][model_key] = {
                "model_label": model_label,
                "standard": round_half_up(scores["baseline/standard"]),
                "end_to_end": round_half_up(scores["baseline/direct"]),
                "context": round_half_up(scores["baseline/context"]),
            }
        rows.append(row)
    return rows


def write_json(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def fmt_num(value: float) -> str:
    return f"{value:.2f}"


def bestify(a: float, b: float, c: float) -> Tuple[str, str, str]:
    best = max(a, b, c)
    vals = []
    for x in (a, b, c):
        text = fmt_num(x)
        vals.append(f"\\textbf{{{text}}}" if x == best else text)
    return tuple(vals)


def write_latex(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[h]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllccccccccc}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{\\textbf{Llama-3.1-8B-Instruct}} & \\multicolumn{3}{c}{\\textbf{Mistral-7B-Instruct-v0.3}} & \\multicolumn{3}{c}{\\textbf{GPT-4o-mini}}  \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
        "\\textbf{Dataset} & \\textbf{Task type} & \\textbf{Metric} & \\cstd & \\etoe & \\cctx & \\cstd & \\etoe & \\cctx & \\cstd & \\etoe & \\cctx \\\\",
        "\\midrule",
    ]
    for row in rows:
        llama = row["models"]["llama"]
        mistral = row["models"]["mistral"]
        gpt = row["models"]["gpt"]
        l_std, l_e2e, l_ctx = bestify(llama["standard"], llama["end_to_end"], llama["context"])
        m_std, m_e2e, m_ctx = bestify(mistral["standard"], mistral["end_to_end"], mistral["context"])
        g_std, g_e2e, g_ctx = bestify(gpt["standard"], gpt["end_to_end"], gpt["context"])
        metric_label = "chrF" if row["metric"] == "chrf" else row["metric"]
        lines.append(
            f'{row["dataset"]} & {row["task_type"]} & {metric_label} & '
            f"{l_std} & {l_e2e} & {l_ctx} & "
            f"{m_std} & {m_e2e} & {m_ctx} & "
            f"{g_std} & {g_e2e} & {g_ctx} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Macro-average performance over languages on open-ended datasets.}",
            "\\end{table*}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def gap_closed_class(value: float) -> str:
    if value >= 0.75:
        return "posstrong"
    if value >= 0.35:
        return "posmid"
    return "posweak"


def write_gap_closed_json(rows: List[Dict], out_path: Path) -> None:
    payload = []
    order = {name: i for i, name in enumerate(GAP_CLOSED_ORDER)}
    for row in sorted(rows, key=lambda r: order.get(r["dataset"], 999)):
        gpt_std = row["models"]["gpt"]["standard"]
        item = {
            "dataset_key": row["dataset_key"],
            "dataset": row["dataset"],
            "metric": "chrF" if row["metric"] == "chrf" else row["metric"],
            "gpt_standard": gpt_std,
            "models": {},
        }
        for model_key in ("llama", "mistral"):
            model = row["models"][model_key]
            numerator = model["context"] - model["standard"]
            denominator = gpt_std - model["standard"]
            gap_closed = numerator / denominator if denominator != 0 else float("nan")
            item["models"][model_key] = {
                "context": model["context"],
                "gap_closed_fraction": gap_closed,
                "gap_closed_percent": int(round_half_up(100.0 * gap_closed, 0)),
            }
        payload.append(item)
    write_json(payload, out_path)


def write_gap_closed_latex(rows: List[Dict], out_path: Path) -> None:
    order = {name: i for i, name in enumerate(GAP_CLOSED_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r["dataset"], 999))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{l c c c c}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Llama-3.1-8B-Instruct}} & \\multicolumn{2}{c}{\\textbf{Mistral-7B-Instruct-v0.3}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "\\textbf{Dataset} & \\cctx & gap closed & \\cctx & gap closed \\\\",
        "\\midrule",
    ]
    for row in rows:
        gpt_std = row["models"]["gpt"]["standard"]
        llama = row["models"]["llama"]
        mistral = row["models"]["mistral"]

        llama_gap = (llama["context"] - llama["standard"]) / (gpt_std - llama["standard"])
        mistral_gap = (mistral["context"] - mistral["standard"]) / (gpt_std - mistral["standard"])

        llama_pct = int(round_half_up(100.0 * llama_gap, 0))
        mistral_pct = int(round_half_up(100.0 * mistral_gap, 0))

        llama_gap_tex = "\\cellcolor{%s}%s%s" % (
            gap_closed_class(llama_gap),
            "\\textbf{" if gap_closed_class(llama_gap) == "posstrong" else "",
            str(llama_pct) + "\\%" + ("}" if gap_closed_class(llama_gap) == "posstrong" else ""),
        )
        mistral_gap_tex = "\\cellcolor{%s}%s%s" % (
            gap_closed_class(mistral_gap),
            "\\textbf{" if gap_closed_class(mistral_gap) == "posstrong" else "",
            str(mistral_pct) + "\\%" + ("}" if gap_closed_class(mistral_gap) == "posstrong" else ""),
        )

        lines.append(
            "%s & $%.2f$ & %s & $%.2f$ & %s \\\\"
            % (
                row["dataset"],
                llama["context"],
                llama_gap_tex,
                mistral["context"],
                mistral_gap_tex,
            )
        )
    gpt_scores = {
        row["dataset"]: row["models"]["gpt"]["standard"]
        for row in rows
    }
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Fraction of the open-model-to-GPT-4o-mini gap closed by the context-aware cascade, on the four open-ended benchmarks. \\emph{Gap closed} is $(\\text{\\cctx} - \\text{\\cstd})/(\\text{GPT-4o-mini}_{\\text{std}} - \\text{\\cstd})$, the share of the standard-cascade gap from the open model to GPT-4o-mini that the context-aware cascade recovers. GPT-4o-mini standard-cascade reference scores: Aya $%.2f$, BLEnD $%.2f$, Global-PIQA-OE $%.2f$, MKQA $%.2f$ (chrF).}"
            % (
                gpt_scores["Aya"],
                gpt_scores["BLEnD"],
                gpt_scores["Global-PIQA-OE"],
                gpt_scores["MKQA"],
            ),
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the open-ended macro-average table from results metrics.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/open_ended_macro_table")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    rows = build_rows(results_root)
    write_json(rows, out_dir / "open_ended_macro_averages.json")
    write_latex(rows, out_dir / "open_ended_macro_averages.tex")
    write_gap_closed_json(rows, out_dir / "open_ended_gap_closed.json")
    write_gap_closed_latex(rows, out_dir / "open_ended_gap_closed.tex")


if __name__ == "__main__":
    main()
