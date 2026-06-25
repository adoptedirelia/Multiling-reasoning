"""
vLLM-based LLM Judge: uses a large model to evaluate whether each prediction
is correct / acceptable given the question and ground truth.

Outputs a JSON file with per-sample judgments and aggregate scores.
"""

import json
import os
import re
import glob
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional

from vllm import LLM, SamplingParams


JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator. Your task is to judge whether a candidate answer is correct or acceptable for a given question, by comparing it with the reference (gold) answer(s).

Scoring criteria:
- 2 = Correct: The candidate answer is essentially correct and conveys the same meaning as the gold answer.
- 1 = Partially correct: The candidate answer captures some relevant information but is incomplete or slightly inaccurate.
- 0 = Incorrect: The candidate answer is wrong, irrelevant, or misses the key point entirely.

You MUST output ONLY a JSON object with two fields:
{"score": <0|1|2>, "reason": "<brief explanation in English>"}

Do NOT output anything else."""


def build_judge_prompt(question: str, gold: str, prediction: str, language: str) -> str:
    return f"""Question ({language}):
{question}

Reference answer:
{gold}

Candidate answer:
{prediction}

Judge the candidate answer. Output ONLY a JSON object: {{"score": <0|1|2>, "reason": "<brief explanation>"}}"""


def parse_judge_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        m = re.search(r'\{[^}]*"score"\s*:\s*(\d)[^}]*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            return {"score": int(obj.get("score", 0)), "reason": obj.get("reason", "")}
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r'[012]', text)
    score = int(m.group(0)) if m else 0
    return {"score": score, "reason": text[:200]}


def _normalize_gts(gts) -> List[str]:
    if isinstance(gts, str):
        return [gts]
    if isinstance(gts, list):
        return [str(g) for g in gts if g is not None]
    return [str(gts)]


def collect_judge_items(path: str) -> List[Dict[str, Any]]:
    """Read a result file and return a flat list of items to judge."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    basename = os.path.basename(path).replace(".json", "")
    is_end_to_end = basename.startswith("end_to_end")

    items = []
    for rec in data:
        question = rec.get("question", "")
        language = rec.get("language", "unknown")
        gts = _normalize_gts(rec.get("ground_truth", []))
        gold_text = " | ".join(gts) if gts else ""
        sample_idx = rec.get("sample_idx", 0)

        if is_end_to_end:
            ans = (rec.get("result") or {}).get("answer", "") or ""
            items.append({
                "file": basename,
                "sample_idx": sample_idx,
                "answer_type": "result",
                "question": question,
                "language": language,
                "gold": gold_text,
                "prediction": ans,
            })
        else:
            llm_ans = (rec.get("llm_result") or {}).get("answer", "") or ""
            mt2_ans = (rec.get("mt2_result") or {}).get("answer", "") or ""
            items.append({
                "file": basename,
                "sample_idx": sample_idx,
                "answer_type": "llm",
                "question": question,
                "language": language,
                "gold": gold_text,
                "prediction": llm_ans,
            })
            items.append({
                "file": basename,
                "sample_idx": sample_idx,
                "answer_type": "mt2",
                "question": question,
                "language": language,
                "gold": gold_text,
                "prediction": mt2_ans,
            })

    return items


def run_judge(
    llm: LLM,
    sampling_params: SamplingParams,
    items: List[Dict[str, Any]],
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    """Run the LLM judge on all items and return items with judgment attached."""
    all_prompts = []
    for item in items:
        prompt = build_judge_prompt(
            question=item["question"],
            gold=item["gold"],
            prediction=item["prediction"],
            language=item["language"],
        )
        all_prompts.append(prompt)

    all_judgments = []
    for start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[start:start + batch_size]

        conversations = [
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in batch_prompts
        ]

        outputs = llm.chat(conversations, sampling_params=sampling_params)

        for output in outputs:
            text = output.outputs[0].text
            judgment = parse_judge_output(text)
            all_judgments.append(judgment)

        print(f"  Judged {min(start + batch_size, len(all_prompts))}/{len(all_prompts)} items")

    for item, judgment in zip(items, all_judgments):
        item["judge_score"] = judgment["score"]
        item["judge_reason"] = judgment["reason"]

    return items


def compute_aggregate(judged_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate scores per file / answer_type / language."""
    groups: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for item in judged_items:
        groups[item["file"]][item["answer_type"]][item["language"]].append(
            item["judge_score"]
        )

    result = {}
    for file_name, by_type in sorted(groups.items()):
        file_result = {}
        for ans_type, by_lang in sorted(by_type.items()):
            type_result = {}
            all_scores = []
            for lang, scores in sorted(by_lang.items()):
                avg = sum(scores) / len(scores) if scores else 0
                type_result[lang] = {
                    "count": len(scores),
                    "avg_score": round(avg, 4),
                    "correct_pct": round(100.0 * sum(1 for s in scores if s == 2) / max(1, len(scores)), 2),
                    "partial_pct": round(100.0 * sum(1 for s in scores if s == 1) / max(1, len(scores)), 2),
                    "incorrect_pct": round(100.0 * sum(1 for s in scores if s == 0) / max(1, len(scores)), 2),
                }
                all_scores.extend(scores)
            overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
            type_result["__overall__"] = {
                "count": len(all_scores),
                "avg_score": round(overall_avg, 4),
                "correct_pct": round(100.0 * sum(1 for s in all_scores if s == 2) / max(1, len(all_scores)), 2),
                "partial_pct": round(100.0 * sum(1 for s in all_scores if s == 1) / max(1, len(all_scores)), 2),
                "incorrect_pct": round(100.0 * sum(1 for s in all_scores if s == 0) / max(1, len(all_scores)), 2),
            }
            file_result[ans_type] = type_result
        result[file_name] = file_result

    return result


def print_summary(aggregate: Dict[str, Any]):
    print("\n" + "=" * 100)
    print("LLM JUDGE SUMMARY")
    print("=" * 100)

    for file_name, by_type in sorted(aggregate.items()):
        print(f"\n--- {file_name} ---")
        for ans_type, by_lang in sorted(by_type.items()):
            print(f"  [{ans_type}]")
            header = f"    {'Language':<15} {'Count':>6} {'AvgScore':>10} {'Correct%':>10} {'Partial%':>10} {'Wrong%':>10}"
            print(header)
            print("    " + "-" * (len(header) - 4))
            for lang, metrics in sorted(by_lang.items()):
                if lang == "__overall__":
                    continue
                print(
                    f"    {lang:<15} {metrics['count']:>6} {metrics['avg_score']:>10.4f}"
                    f" {metrics['correct_pct']:>10.2f} {metrics['partial_pct']:>10.2f}"
                    f" {metrics['incorrect_pct']:>10.2f}"
                )
            ov = by_lang["__overall__"]
            print(
                f"    {'OVERALL':<15} {ov['count']:>6} {ov['avg_score']:>10.4f}"
                f" {ov['correct_pct']:>10.2f} {ov['partial_pct']:>10.2f}"
                f" {ov['incorrect_pct']:>10.2f}"
            )

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="vLLM-based LLM Judge for evaluation results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data_transfer/llama_result",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Single result file to judge (overrides --input_dir)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Judge model name or path",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens for judge output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/llm_judge_results.json",
        help="Path to save detailed judgment results",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="./results/llm_judge_summary.json",
        help="Path to save aggregate summary",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Max model context length for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    args = parser.parse_args()

    if args.input_file:
        json_files = [args.input_file]
    else:
        json_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found")
        return

    print(f"Collecting items to judge from {len(json_files)} files...")
    all_items = []
    for path in json_files:
        items = collect_judge_items(path)
        all_items.extend(items)
        print(f"  {os.path.basename(path)}: {len(items)} items")

    print(f"\nTotal items to judge: {len(all_items)}")

    print(f"\nLoading vLLM model: {args.model}")
    print(f"  tensor_parallel_size={args.tensor_parallel_size}")
    print(f"  max_model_len={args.max_model_len}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=1.0 if args.temperature == 0.0 else 0.95,
    )

    print("\nRunning LLM judge...")
    judged_items = run_judge(llm, sampling_params, all_items, batch_size=args.batch_size)

    aggregate = compute_aggregate(judged_items)
    print_summary(aggregate)

    output_dir = os.path.dirname(args.output) or "."
    summary_dir = os.path.dirname(args.output_summary) or "."
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    items_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in judged_items:
        items_by_file[item["file"]].append(item)

    for file_name, file_items in items_by_file.items():
        per_file_path = os.path.join(output_dir, f"llm_judge_{file_name}.json")
        with open(per_file_path, "w", encoding="utf-8") as f:
            json.dump(file_items, f, ensure_ascii=False, indent=2)
        print(f"  {file_name}: {len(file_items)} items -> {per_file_path}")

    for file_name, file_agg in aggregate.items():
        per_summary_path = os.path.join(summary_dir, f"llm_judge_summary_{file_name}.json")
        with open(per_summary_path, "w", encoding="utf-8") as f:
            json.dump({file_name: file_agg}, f, ensure_ascii=False, indent=2)

    print(f"\nPer-file detailed judgments saved to {output_dir}/llm_judge_<name>.json")
    print(f"Per-file summaries saved to {summary_dir}/llm_judge_summary_<name>.json")


if __name__ == "__main__":
    main()
