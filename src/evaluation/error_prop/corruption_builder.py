import argparse
import logging
import os
import random
from typing import Dict, List

from .config import ErrorSimConfig, load_config
from .corruptions_input import generate_input_errors
from .corruptions_output import apply_repeated_corruption
from .corruption_data import write_corruptions_jsonl
from .engine_factory import create_engine
from .mkqa_loader import load_mkqa_records
from .models import LLMInputCorruptor, LLMOutputCorruptor


def _engine_key(cfg):
    return (
        cfg.model_type,
        cfg.model_name,
        cfg.device_map,
        cfg.torch_dtype,
        cfg.attn_implementation,
        tuple(sorted(cfg.engine_kwargs.items())),
    )


def build_corruptions_mkqa(config_path: str) -> str:
    cfg: ErrorSimConfig = load_config(config_path)
    if not cfg.corruption.output_jsonl:
        raise ValueError("corruption.output_jsonl must be set in config to write corruptions.")

    logging.basicConfig(level=logging.INFO)

    rng = random.Random(cfg.dataset.seed)

    # engines (reuse if same model)
    engine_cache = {}

    def get_engine(mcfg):
        key = _engine_key(mcfg)
        if key in engine_cache:
            return engine_cache[key]
        eng = create_engine(mcfg)
        eng.load_model()
        engine_cache[key] = eng
        return eng

    corruption_engine = get_engine(cfg.models.get("corruption_llm", cfg.models["reasoner_en"]))

    input_corruptor = LLMInputCorruptor(corruption_engine)
    output_corruptor = LLMOutputCorruptor(corruption_engine)

    base_lang = cfg.dataset.langs[0]
    records = load_mkqa_records(cfg.dataset.mkqa_path, base_lang, cfg.dataset.max_examples)
    if not records:
        raise ValueError(f"No records found for base language {base_lang}")

    example_ids = [r["example_id"] for r in records]
    x_en = [r["x_en"] for r in records]
    y_en_gold = [
        " ||| ".join(r.get("y_en_gold") or [""]) if r.get("y_en_gold") else ""
        for r in records
    ]

    rows: List[Dict] = []

    # rule-based input errors
    for i, q in enumerate(x_en):
        variants = generate_input_errors(q, rng, cfg.corruption.omission_max_words)
        for err_type, x_err in variants:
            rows.append(
                {
                    "example_id": example_ids[i],
                    "error_group": "input_err",
                    "error_type": err_type,
                    "x_en": q,
                    "x_en_err": x_err,
                    "r_en_err": None,
                    "y_en_err": None,
                }
            )

    # LLM-based input errors
    for err_type in ["shift_intent"]:
        x_en_errs = input_corruptor.corrupt_batch(
            x_en,
            err_type,
            max_new_tokens=cfg.generation.corruption_max_new_tokens,
            temperature=cfg.generation.corruption_temperature,
            top_p=cfg.generation.corruption_top_p,
        )
        for i, x_err in enumerate(x_en_errs):
            rows.append(
                {
                    "example_id": example_ids[i],
                    "error_group": "input_err",
                    "error_type": err_type,
                    "x_en": x_en[i],
                    "x_en_err": x_err,
                    "r_en_err": None,
                    "y_en_err": None,
                }
            )

    # LLM-based output errors
    for err_type in ["assumption", "inconsistency", "drift"]:
        corrupted = output_corruptor.corrupt_batch(
            x_en,
            ["" for _ in x_en],
            y_en_gold,
            err_type,
            max_new_tokens=cfg.generation.corruption_max_new_tokens,
            temperature=cfg.generation.corruption_temperature,
            top_p=cfg.generation.corruption_top_p,
        )
        for i, out in enumerate(corrupted):
            r_err = out.get("r_en_err")
            y_err = out.get("y_en_err")
            rows.append(
                {
                    "example_id": example_ids[i],
                    "error_group": "output_err",
                    "error_type": err_type,
                    "x_en": x_en[i],
                    "x_en_err": None,
                    "r_en_err": r_err,
                    "y_en_err": y_err,
                }
            )

    write_corruptions_jsonl(cfg.corruption.output_jsonl, rows)
    return cfg.corruption.output_jsonl


def main():
    ap = argparse.ArgumentParser(description="Build English-side corruptions for MKQA")
    ap.add_argument("--config", required=True, help="Path to error simulation config JSON")
    args = ap.parse_args()
    out_path = build_corruptions_mkqa(args.config)
    logging.info("Wrote corruptions to %s", out_path)


if __name__ == "__main__":
    main()
