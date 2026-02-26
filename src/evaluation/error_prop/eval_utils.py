import json
import os
import subprocess
from typing import Dict, List

import importlib.util


def _load_mkqa_eval_util(mkqa_dir: str):
    path = os.path.join(mkqa_dir, "mkqa_eval_util.py")
    spec = importlib.util.spec_from_file_location("mkqa_eval_util", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load mkqa_eval_util from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_predictions_jsonl(path: str, preds: Dict[str, str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex_id, pred in preds.items():
            rec = {
                "example_id": ex_id,
                "prediction": pred,
                "binary_answer": None,
                "no_answer_prob": 0.0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_mkqa_eval(mkqa_dir: str, annotation_file: str, predictions_file: str, language: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "python",
        os.path.join(mkqa_dir, "mkqa_eval.py"),
        "--annotation_file",
        annotation_file,
        "--predictions_file",
        predictions_file,
        "--language",
        language,
        "--out-dir",
        out_dir,
    ]
    env = os.environ.copy()
    env["MKQA_NO_PLOTS"] = "1"
    subprocess.check_call(cmd, env=env)


def run_simple_eval(
    predictions: Dict[str, str],
    gold_answers: Dict[str, List[str]],
    language: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    mkqa_eval_util = _load_mkqa_eval_util(os.path.join(os.path.dirname(__file__), "ml-mkqa"))
    em_scores = {}
    f1_scores = {}
    for ex_id, pred in predictions.items():
        golds = gold_answers.get(ex_id, [])
        if not golds:
            continue
        em = mkqa_eval_util.compute_max_score_over_answers(
            mkqa_eval_util.calculate_em, pred, golds, language
        )
        f1 = mkqa_eval_util.compute_max_score_over_answers(
            mkqa_eval_util.calculate_f1, pred, golds, language
        )
        em_scores[ex_id] = em
        f1_scores[ex_id] = f1

    metrics = {
        "exact_match": round(100.0 * (sum(em_scores.values()) / max(1, len(em_scores))), 2),
        "f1": round(100.0 * (sum(f1_scores.values()) / max(1, len(f1_scores))), 2),
        "num_examples": len(em_scores),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
