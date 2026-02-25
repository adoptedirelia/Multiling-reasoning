import json
import argparse
import string
from collections import Counter
from statistics import mean


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

    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_toks)
    recall = num_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def max_over_answers(metric_fn, pred: str, gold_list):
    if not gold_list:
        return 0.0
    return max(metric_fn(pred, g) for g in gold_list)


def evaluate_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_em_list, llm_f1_list = [], []
    mt2_em_list, mt2_f1_list = [], []

    for rec in data:
        gts = rec.get("ground_truth", [])
        if isinstance(gts, str):
            gts = [gts]
        elif not isinstance(gts, list):
            gts = [str(gts)]

        llm_ans = (rec.get("llm_result") or {}).get("answer", "") or ""
        mt2_ans = (rec.get("mt2_result") or {}).get("answer", "") or ""

        llm_em = max_over_answers(exact_match, llm_ans, gts)
        llm_f1 = max_over_answers(f1_score, llm_ans, gts)
        llm_em_list.append(llm_em)
        llm_f1_list.append(llm_f1)

        mt2_em = max_over_answers(exact_match, mt2_ans, gts)
        mt2_f1 = max_over_answers(f1_score, mt2_ans, gts)
        mt2_em_list.append(mt2_em)
        mt2_f1_list.append(mt2_f1)

    def pct(x):
        return round(100.0 * x, 2)

    metrics = {
        "num_samples": len(data),
        "llm_exact_match": pct(mean(llm_em_list)) if llm_em_list else 0.0,
        "llm_f1": pct(mean(llm_f1_list)) if llm_f1_list else 0.0,
        "mt2_exact_match": pct(mean(mt2_em_list)) if mt2_em_list else 0.0,
        "mt2_f1": pct(mean(mt2_f1_list)) if mt2_f1_list else 0.0,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="./results/cascade_piqa_lora.json",
        help="path to output_error_cascade_mkqa.json",
    )
    args = parser.parse_args()
    evaluate_file(args.input)


if __name__ == "__main__":
    main()