import argparse
import json
import logging
import os
import random
from typing import Dict, Iterable, List

from ..config import V2Config, load_config
from ..loaders.registry import load_records_by_language
from ..runtime.corruptions_input import generate_input_errors
from ..runtime.engine_factory import create_engine
from ..runtime.models import ENReasoner, LLMInputCorruptor, LLMOutputCorruptor, MT1Translator

LOGGER = logging.getLogger(__name__)

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


def _engine_key(cfg):
    try:
        engine_kwargs_key = json.dumps(cfg.engine_kwargs, sort_keys=True, default=str)
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


def _write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_corruptions(config_path: str) -> str:
    cfg: V2Config = load_config(config_path)
    out_path = cfg.corruption.output_jsonl or cfg.outputs.corruption_jsonl
    if not out_path:
        raise ValueError("Missing corruption output path.")

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("build_corruptions start: run_name=%s dataset=%s", cfg.run_name, cfg.dataset.dataset_type)
    LOGGER.info("build_corruptions output=%s langs=%s", out_path, cfg.dataset.langs)
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
        reasoner = ENReasoner(get_engine(cfg.models["reasoner_en"]))
        mt1_engine = get_engine(cfg.models["mt1"])
        corruption_engine = get_engine(cfg.models["corruption_llm"])
        input_corruptor = LLMInputCorruptor(corruption_engine)
        output_corruptor = LLMOutputCorruptor(corruption_engine)

        rows: List[Dict] = []
        by_lang = load_records_by_language(cfg.dataset)
        LOGGER.info("loaded records by language: %s", {k: len(v) for k, v in by_lang.items()})
        for lang in cfg.dataset.langs:
            records = by_lang.get(lang, [])
            if not records:
                LOGGER.info("lang=%s skipped: no records", lang)
                continue
            LOGGER.info("lang=%s build start n=%d", lang, len(records))
            mt1 = MT1Translator(mt1_engine, src_lang=_LANG_DISPLAY_NAME.get(lang, lang))

            example_ids = [r["example_id"] for r in records]
            x_l = [r["x_l"] for r in records]
            x_en = mt1.run_batch(
                x_l,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            x_en = [xe.strip() if isinstance(xe, str) else "" for xe in x_en]

            base = reasoner.run_batch(
                x_en,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            r_en_base = [b["reasoning"] for b in base]
            y_en_base = [b["answer"] for b in base]

            for i, q in enumerate(x_en):
                variants = generate_input_errors(q, rng, cfg.corruption.omission_max_words)
                for err_type, x_err in variants:
                    rows.append(
                        {
                            "dataset": cfg.dataset.dataset_type,
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
            LOGGER.info("lang=%s added deterministic input errors=%d", lang, len(example_ids) * 2)

            x_en_errs = input_corruptor.corrupt_batch(
                x_en,
                "shift_intent",
                max_new_tokens=cfg.generation.corruption_max_new_tokens,
                temperature=cfg.generation.corruption_temperature,
                top_p=cfg.generation.corruption_top_p,
            )
            for i, x_err in enumerate(x_en_errs):
                rows.append(
                    {
                        "dataset": cfg.dataset.dataset_type,
                        "lang": lang,
                        "example_id": example_ids[i],
                        "error_group": "input_err",
                        "error_type": "shift_intent",
                        "x_en": x_en[i],
                        "x_en_err": x_err,
                        "r_en_err": None,
                        "y_en_err": None,
                    }
                )
            LOGGER.info("lang=%s added llm input errors=%d", lang, len(x_en_errs))

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
                            "dataset": cfg.dataset.dataset_type,
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
                LOGGER.info("lang=%s added output errors type=%s n=%d", lang, err_type, len(corrupted))

            LOGGER.info("lang=%s build done cumulative_rows=%d", lang, len(rows))

        _write_jsonl(out_path, rows)
        LOGGER.info("build_corruptions done: wrote_rows=%d path=%s", len(rows), out_path)
        return out_path
    finally:
        for eng in engine_cache.values():
            try:
                eng.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Build v2 corruptions")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    out = build_corruptions(args.config)
    if args.output and args.output != out:
        raise ValueError("Use config corruption.output_jsonl or outputs.corruption_jsonl for output path.")


if __name__ == "__main__":
    main()
