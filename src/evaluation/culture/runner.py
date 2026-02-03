import json
import os
from typing import Any, Dict, List, Literal

from .client import LLMClient
from .config import CultureEvalConfig, load_config
from .prompts import (
    build_answer_eval_prompt,
    build_ground_truth_eval_prompt,
    build_answer_with_ground_truth_eval_prompt,
)


def _extract_answer(record: Dict[str, Any]) -> str:
    mt2 = record.get("mt2_result", "")
    if isinstance(mt2, dict):
        return str(mt2.get("answer", "")).strip()
    return str(mt2).strip()


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _save_results(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def run(config_path: str, mode: Literal["answer", "ground_truth", "answer_with_gt"]) -> None:
    cfg: CultureEvalConfig = load_config(config_path)
    records = _load_records(cfg.input_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]

    clients = [
        LLMClient(
            provider=p.provider,
            model=p.model,
            api_key_env=p.api_key_env,
            timeout_s=cfg.request_timeout_s,
        )
        for p in cfg.providers
    ]

    results: List[Dict[str, Any]] = []
    for rec in records:
        question = str(rec.get("question", "")).strip()
        ground_truth = str(rec.get("ground_truth", "")).strip()
        answer = _extract_answer(rec)

        row = {
            "sample_idx": rec.get("sample_idx"),
            "language": rec.get("language"),
            "question": question,
        }

        if mode == "answer":
            row["mt2_answer"] = answer
            prompt = build_answer_eval_prompt(question, answer)
        elif mode == "ground_truth":
            row["ground_truth"] = ground_truth
            prompt = build_ground_truth_eval_prompt(question, ground_truth)
        else:
            row["ground_truth"] = ground_truth
            row["mt2_answer"] = answer
            prompt = build_answer_with_ground_truth_eval_prompt(question, ground_truth, answer)

        for p_cfg, client in zip(cfg.providers, clients):
            scores = client.score(
                prompt, temperature=p_cfg.temperature, max_output_tokens=p_cfg.max_output_tokens
            )
            row[f"{p_cfg.provider}_culture"] = scores["culture"]
            row[f"{p_cfg.provider}_correctness"] = scores["correctness"]

        results.append(row)

    _save_results(cfg.output_path, results)
