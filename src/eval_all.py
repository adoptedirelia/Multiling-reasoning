"""
Batch evaluation script for all result files under data_transfer/llama_result/.
Computes per-language EM / F1 and prints a summary table.
"""

import json
import os
import glob
import string
import argparse
from collections import defaultdict
from statistics import mean
from typing import List, Dict, Any


def normalize(text: str) -> str:
    if text is None:
        text = ""
    text = text.lower().strip()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)
    pred_toks = pred_norm.split() if pred_norm else []
    gold_toks = gold_norm.split() if gold_norm else []
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    from collections import Counter
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_toks)
    recall = num_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_answers(metric_fn, pred: str, gold_list: List[str]) -> float:
    if not gold_list:
        return 0.0
    return max(metric_fn(pred, g) for g in gold_list if g is not None)


def _normalize_gts(gts) -> List[str]:
    if isinstance(gts, str):
        return [gts]
    if isinstance(gts, list):
        return [str(g) for g in gts if g is not None]
    return [str(gts)]


def extract_answers(rec: Dict[str, Any], file_type: str):
    """Return (gts, answer_dict) where answer_dict maps answer_name -> answer_str."""
    gts = _normalize_gts(rec.get("ground_truth", []))
    answers = {}

    if file_type == "end_to_end":
        ans = (rec.get("result") or {}).get("answer", "") or ""
        answers["result"] = ans
    else:
        llm_ans = (rec.get("llm_result") or {}).get("answer", "") or ""
        mt2_ans = (rec.get("mt2_result") or {}).get("answer", "") or ""
        answers["llm"] = llm_ans
        answers["mt2"] = mt2_ans

    return gts, answers


def detect_file_type(filename: str) -> str:
    base = os.path.basename(filename).replace(".json", "")
    if base.startswith("end_to_end"):
        return "end_to_end"
    return "cascade_or_prompting"


def evaluate_single_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_type = detect_file_type(path)

    metrics_by_lang: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in data:
        language = rec.get("language", "unknown")
        gts, answers = extract_answers(rec, file_type)
        if not gts:
            continue

        for ans_name, ans_text in answers.items():
            em = max_over_answers(exact_match, ans_text, gts)
            f1 = max_over_answers(f1_score, ans_text, gts)
            metrics_by_lang[language][f"{ans_name}_em"].append(em)
            metrics_by_lang[language][f"{ans_name}_f1"].append(f1)

    result = {
        "file": os.path.basename(path),
        "file_type": file_type,
        "num_samples": len(data),
        "by_language": {},
    }

    pct = lambda x: round(100.0 * x, 2)

    all_metrics_flat: Dict[str, List[float]] = defaultdict(list)

    for lang, lang_metrics in sorted(metrics_by_lang.items()):
        lang_result = {"count": len(next(iter(lang_metrics.values())))}
        for metric_name, values in sorted(lang_metrics.items()):
            lang_result[metric_name] = pct(mean(values))
            all_metrics_flat[metric_name].extend(values)
        result["by_language"][lang] = lang_result

    result["overall"] = {
        metric_name: pct(mean(values))
        for metric_name, values in sorted(all_metrics_flat.items())
    }

    return result


def print_table(all_results: List[Dict[str, Any]]):
    print("\n" + "=" * 120)
    print("EVALUATION SUMMARY")
    print("=" * 120)

    for res in all_results:
        print(f"\n--- {res['file']} (type={res['file_type']}, n={res['num_samples']}) ---")

        langs = sorted(res["by_language"].keys())
        metric_names = sorted(
            k for k in next(iter(res["by_language"].values())).keys() if k != "count"
        )

        header = f"{'Language':<15} {'Count':>6}"
        for m in metric_names:
            header += f"  {m:>12}"
        print(header)
        print("-" * len(header))

        for lang in langs:
            row = res["by_language"][lang]
            line = f"{lang:<15} {row['count']:>6}"
            for m in metric_names:
                line += f"  {row[m]:>12.2f}"
            print(line)

        overall = res["overall"]
        line = f"{'OVERALL':<15} {res['num_samples']:>6}"
        for m in metric_names:
            line += f"  {overall[m]:>12.2f}"
        print(line)

    print("\n" + "=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all result files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data_transfer/llama_result",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON",
    )
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    print(f"Found {len(json_files)} files to evaluate:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")

    all_results = []
    for path in json_files:
        print(f"\nEvaluating {os.path.basename(path)}...")
        result = evaluate_single_file(path)
        all_results.append(result)

    print_table(all_results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
