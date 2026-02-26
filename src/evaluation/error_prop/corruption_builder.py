import argparse
import logging
import random
import json as _json
import threading
from typing import Dict, List

from .config import ErrorSimConfig, load_config
from .corruptions_input import generate_input_errors
from .corruption_data import write_corruptions_jsonl
from .dataset_loader import load_records_by_language
from .engine_factory import create_engine
from .models import ENReasoner, LLMInputCorruptor, LLMOutputCorruptor, MT1Translator


_LANG_DISPLAY_NAME = {
    "amh": "Amharic",
    "jpn": "Japanese",
    "mar": "Marathi",
    "vie": "Vietnamese",
    "zho": "Chinese",
    "arb": "Arabic",
    "tel": "Telugu",
    "zh_cn": "Chinese",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "te": "Telugu",
    "mr": "Marathi",
    "am": "Amharic",
}


def _shutdown_engine_with_timeout(eng, timeout_s: float = 15.0) -> bool:
    """Best-effort shutdown without allowing hangs at process end."""
    done = {"ok": False}

    def _run():
        try:
            eng.shutdown()
        except Exception:
            pass
        finally:
            done["ok"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return done["ok"]


def _engine_key(cfg):
    try:
        engine_kwargs_key = _json.dumps(cfg.engine_kwargs, sort_keys=True, default=str)
    except TypeError:
        engine_kwargs_key = repr(cfg.engine_kwargs)
    return (
        cfg.model_type,
        cfg.model_name,
        cfg.device_map,
        cfg.torch_dtype,
        cfg.attn_implementation,
        engine_kwargs_key,
    )


def build_corruptions(config_path: str) -> str:
    cfg: ErrorSimConfig = load_config(config_path)
    if not cfg.corruption.output_jsonl:
        raise ValueError("corruption.output_jsonl must be set in config to write corruptions.")

    logging.basicConfig(level=logging.INFO)
    rng = random.Random(cfg.dataset.seed)

    engine_cache = {}

    def get_engine(mcfg):
        key = _engine_key(mcfg)
        if key in engine_cache:
            return engine_cache[key]
        eng = create_engine(mcfg)
        eng.load_model()
        engine_cache[key] = eng
        return eng

    try:
        reasoner_engine = get_engine(cfg.models["reasoner_en"])
        mt1_engine = get_engine(cfg.models.get("mt1", cfg.models["reasoner_en"]))
        corruption_engine = get_engine(cfg.models.get("corruption_llm", cfg.models["reasoner_en"]))

        reasoner = ENReasoner(reasoner_engine)
        input_corruptor = LLMInputCorruptor(corruption_engine)
        output_corruptor = LLMOutputCorruptor(corruption_engine)

        target_examples = cfg.dataset.max_examples
        records_by_lang = load_records_by_language(cfg.dataset, max_examples_override=target_examples)
        rows: List[Dict] = []

        for lang in cfg.dataset.langs:
            mt1_translator = MT1Translator(mt1_engine, src_lang=_LANG_DISPLAY_NAME.get(lang, lang))
            records = records_by_lang.get(lang, [])
            if not records:
                logging.warning("No records for language=%s; skipping", lang)
                continue
            records = records[:target_examples]
            logging.info(
                "Building corruptions for language=%s using=%d target(max)=%d",
                lang,
                len(records),
                target_examples,
            )

            example_ids = [r["example_id"] for r in records]
            x_l = [r["x_l"] for r in records]

            # Always create x_en through MT1 translation from x_l.
            x_en = mt1_translator.run_batch(
                x_l,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            x_en = [xe.strip() if isinstance(xe, str) else "" for xe in x_en]

            # Baseline English reasoning+answer used to generate output corruptions.
            base = reasoner.run_batch(
                x_en,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            r_en_base = [b["reasoning"] for b in base]
            y_en_base = [b["answer"] for b in base]

            # Rule-based input errors.
            for i, q in enumerate(x_en):
                variants = generate_input_errors(q, rng, cfg.corruption.omission_max_words)
                for err_type, x_err in variants:
                    rows.append(
                        {
                            "lang": lang,
                            "example_id": example_ids[i],
                            "error_group": "input_err",
                            "error_type": err_type,
                            "x_en": q,
                            "x_en_err": x_err,
                            "r_en_err": None,
                            "y_en_err": None,
                        }
                    )

            # LLM-based input errors.
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
                            "lang": lang,
                            "example_id": example_ids[i],
                            "error_group": "input_err",
                            "error_type": err_type,
                            "x_en": x_en[i],
                            "x_en_err": x_err,
                            "r_en_err": None,
                            "y_en_err": None,
                        }
                    )

            # LLM-based output errors from baseline reasoner outputs.
            for err_type in ["contradiction", "invented", "subjective"]:
                corrupted = output_corruptor.corrupt_batch(
                    x_en,
                    r_en_base,
                    y_en_base,
                    err_type,
                    max_new_tokens=cfg.generation.corruption_max_new_tokens,
                    temperature=cfg.generation.corruption_temperature,
                    top_p=cfg.generation.corruption_top_p,
                )
                for i, out in enumerate(corrupted):
                    rows.append(
                        {
                            "lang": lang,
                            "example_id": example_ids[i],
                            "error_group": "output_err",
                            "error_type": err_type,
                            "x_en": x_en[i],
                            "x_en_err": None,
                            "r_en_err": out.get("r_en_err"),
                            "y_en_err": out.get("y_en_err"),
                        }
                    )

        write_corruptions_jsonl(cfg.corruption.output_jsonl, rows)
        logging.info("Corruptions written to %s; starting engine teardown", cfg.corruption.output_jsonl)
        return cfg.corruption.output_jsonl
    finally:
        for idx, eng in enumerate(engine_cache.values()):
            try:
                ok = _shutdown_engine_with_timeout(eng, timeout_s=15.0)
                if not ok:
                    logging.warning("Engine teardown timed out (index=%d); continuing exit", idx)
            except Exception:
                logging.warning("Engine teardown failed (index=%d); continuing exit", idx)


def main():
    ap = argparse.ArgumentParser(description="Build English-side corruptions")
    ap.add_argument("--config", required=True, help="Path to error simulation config JSON")
    args = ap.parse_args()
    out_path = build_corruptions(args.config)
    logging.info("Wrote corruptions to %s", out_path)


if __name__ == "__main__":
    main()
