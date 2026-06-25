import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

from ..config import V2Config, load_config
from ..loaders.registry import load_records_by_language
from ..runtime.engine_factory import create_engine
from ..runtime.language_names import target_language_name
from ..runtime.models import MT1Translator, _invalid_reasoner_pair, _sanitize_answer_fallback, _strip_prompt_echo
from ..runtime.prompts import (
    direct_answer_prompt,
    mt2_answer_plus_english_question_prompt,
    mt2_answer_plus_reasoning_prompt,
    mt2_answer_plus_source_question_prompt,
    mt2_context_prompt,
    mt2_standard_prompt,
    reasoner_prompt,
)
from ..runtime.text_utils import extract_answer, extract_reasoning

LOGGER = logging.getLogger(__name__)
ABLATION_MODES = {
    "answer_plus_source_question",
    "answer_plus_english_question",
    "answer_plus_reasoning",
}
VALID_BASELINE_MODES = {"standard", "context", "direct", *ABLATION_MODES}


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


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _predictions_root(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return "results/v2/raw_preds"
    if p.endswith(".jsonl"):
        return os.path.splitext(p)[0]
    return p


def _mode_list(modes: List[str]) -> List[str]:
    out = []
    for m in modes:
        lm = m.lower()
        if lm == "both":
            out.extend(["standard", "context"])
        else:
            out.append(lm)
    keep = []
    seen = set()
    for m in out:
        if m in VALID_BASELINE_MODES and m not in seen:
            seen.add(m)
            keep.append(m)
    if not keep:
        raise ValueError(
            "No valid cascade modes. Use standard/context/direct/both/"
            "answer_plus_source_question/answer_plus_english_question/answer_plus_reasoning."
        )
    return keep


def _mt2_prompt_for_mode(
    mode: str,
    *,
    x_l: str,
    x_en: str,
    r_en: str,
    y_en: str,
    target_lang: str,
) -> str:
    if mode == "standard":
        return mt2_standard_prompt(y_en, target_lang)
    if mode == "context":
        return mt2_context_prompt(x_l, x_en, r_en, y_en, target_lang)
    if mode == "answer_plus_source_question":
        return mt2_answer_plus_source_question_prompt(x_l, y_en, target_lang)
    if mode == "answer_plus_english_question":
        return mt2_answer_plus_english_question_prompt(x_en, y_en, target_lang)
    if mode == "answer_plus_reasoning":
        return mt2_answer_plus_reasoning_prompt(r_en, y_en, target_lang)
    raise ValueError(f"Unsupported MT2 mode: {mode}")


def _parse_reasoner(raw: str, prompt: str, question_en: str) -> Dict[str, str]:
    body = _strip_prompt_echo(raw, prompt)
    reasoning = extract_reasoning(body)
    answer = extract_answer(body)
    if _invalid_reasoner_pair(reasoning, answer):
        answer = _sanitize_answer_fallback(answer, question_en)
        reasoning = ""
    return {"reasoning": reasoning, "answer": answer}


def run_raw_predictions(
    config_path: str,
    corruption_jsonl: str = "",
    out_jsonl: str = "",
    modes: str = "",
    baseline_only: bool = False,
) -> str:
    cfg: V2Config = load_config(config_path)
    corr_path = corruption_jsonl or cfg.corruption.input_jsonl or cfg.corruption.output_jsonl or cfg.outputs.corruption_jsonl
    out_root = _predictions_root(out_jsonl or cfg.outputs.predictions_jsonl)
    mode_list = _mode_list(modes.split(",") if modes else cfg.cascade.modes)

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("run_raw_predictions start: run_name=%s dataset=%s", cfg.run_name, cfg.dataset.dataset_type)
    rows_by_lang: Dict[str, List[Dict]] = defaultdict(list)
    if not baseline_only:
        if not corr_path:
            raise ValueError("Missing corruption JSONL path unless --baseline-only is set.")
        if not os.path.exists(corr_path):
            raise FileNotFoundError(corr_path)
        LOGGER.info("inputs: corruptions=%s out_root=%s modes=%s", corr_path, out_root, mode_list)
        rows = _load_jsonl(corr_path)
        LOGGER.info("loaded corruption rows=%d", len(rows))
        for r in rows:
            lang = r.get("lang")
            if lang:
                rows_by_lang[lang].append(r)
    else:
        LOGGER.info("inputs: baseline_only=true out_root=%s modes=%s", out_root, mode_list)

    records_by_lang = load_records_by_language(cfg.dataset)
    LOGGER.info("loaded records by language: %s", {k: len(v) for k, v in records_by_lang.items()})

    engine_cache = {}

    def get_engine(mcfg):
        key = _engine_key(mcfg)
        if key in engine_cache:
            return engine_cache[key]
        eng = create_engine(mcfg)
        eng.load_model()
        engine_cache[key] = eng
        return eng

    out_rows_by_lang: Dict[str, List[Dict]] = defaultdict(list)
    try:
        mt1_engine = get_engine(cfg.models["mt1"])
        reasoner_engine = get_engine(cfg.models["reasoner_en"])
        mt2_engine = get_engine(cfg.models["mt2"])
        os.makedirs(out_root, exist_ok=True)

        for lang in cfg.dataset.langs:
            target_lang = target_language_name(lang)
            rec_list = records_by_lang.get(lang, [])
            if not rec_list:
                LOGGER.info("lang=%s skipped: no dataset records", lang)
                continue
            rec_map = {r["example_id"]: r for r in rec_list}
            input_map: Dict[str, List[Dict]] = defaultdict(list)
            output_map: Dict[str, List[Dict]] = defaultdict(list)
            if baseline_only:
                example_ids = [r["example_id"] for r in rec_list]
                if len(example_ids) > cfg.dataset.max_examples:
                    example_ids = example_ids[: cfg.dataset.max_examples]
                if not example_ids:
                    LOGGER.info("lang=%s skipped: no dataset example_ids", lang)
                    continue
                LOGGER.info("lang=%s baseline-only start dataset_records=%d using=%d", lang, len(rec_list), len(example_ids))
                x_l = [rec_map[ex_id]["x_l"] for ex_id in example_ids]
                mt1 = MT1Translator(mt1_engine, src_lang=target_lang)
                x_en = mt1.run_batch(
                    x_l,
                    max_new_tokens=cfg.generation.mt1_max_new_tokens,
                    temperature=cfg.generation.mt1_temperature,
                    top_p=cfg.generation.mt1_top_p,
                )
            else:
                lang_rows = rows_by_lang.get(lang, [])
                if not lang_rows:
                    LOGGER.info("lang=%s skipped: no corruption rows", lang)
                    continue
                LOGGER.info("lang=%s start dataset_records=%d corruption_rows=%d", lang, len(rec_list), len(lang_rows))

                x_en_map = {}
                for r in lang_rows:
                    ex_id = r["example_id"]
                    x_en_map[ex_id] = r.get("x_en") or x_en_map.get(ex_id, "")
                    if r.get("error_group") == "input_err":
                        input_map[ex_id].append(
                            {"error_type": r["error_type"], "x_en_err": r.get("x_en_err")}
                        )
                    elif r.get("error_group") == "output_err":
                        output_map[ex_id].append(
                            {
                                "error_type": r["error_type"],
                                "r_en_err": r.get("r_en_err"),
                                "y_en_err": r.get("y_en_err"),
                            }
                        )

                example_ids = [ex_id for ex_id in x_en_map.keys() if ex_id in rec_map]
                if len(example_ids) > cfg.dataset.max_examples:
                    example_ids = example_ids[: cfg.dataset.max_examples]
                if not example_ids:
                    LOGGER.info("lang=%s skipped: no overlapping example_ids", lang)
                    continue
                LOGGER.info("lang=%s using example_ids=%d", lang, len(example_ids))

                x_l = [rec_map[ex_id]["x_l"] for ex_id in example_ids]
                x_en = [x_en_map[ex_id].strip() for ex_id in example_ids]

            base_reasoner_prompts = [reasoner_prompt(q) for q in x_en]
            base_reasoner_raw = reasoner_engine.generate_batch(
                base_reasoner_prompts,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            base_parsed = [
                _parse_reasoner(base_reasoner_raw[i], base_reasoner_prompts[i], x_en[i])
                for i in range(len(example_ids))
            ]
            r_en_base = [p["reasoning"] for p in base_parsed]
            y_en_base = [p["answer"] for p in base_parsed]

            for mode in mode_list:
                LOGGER.info("lang=%s baseline mode=%s n=%d", lang, mode, len(example_ids))
                if mode != "direct":
                    prompts = [
                        _mt2_prompt_for_mode(
                            mode,
                            x_l=x_l[i],
                            x_en=x_en[i],
                            r_en=r_en_base[i],
                            y_en=y_en_base[i],
                            target_lang=target_lang,
                        )
                        for i in range(len(example_ids))
                    ]
                    raw_mt2 = mt2_engine.generate_batch(
                        prompts,
                        max_new_tokens=cfg.generation.mt2_max_new_tokens,
                        temperature=cfg.generation.mt2_temperature,
                        top_p=cfg.generation.mt2_top_p,
                    )
                    for i, ex_id in enumerate(example_ids):
                        out_rows_by_lang[lang].append(
                            {
                                "dataset": cfg.dataset.dataset_type,
                                "lang": lang,
                                "target_lang": target_lang,
                                "example_id": ex_id,
                                "error_group": "baseline",
                                "error_type": None,
                                "cascade_mode": mode,
                                "slice": f"baseline/{mode}",
                                "x_l": x_l[i],
                                "x_en": x_en[i],
                                "r_en": r_en_base[i],
                                "y_en": y_en_base[i],
                                "x_en_err": None,
                                "r_en_err": None,
                                "y_en_err": None,
                                "reasoner_prompt": base_reasoner_prompts[i],
                                "reasoner_raw": base_reasoner_raw[i],
                                "mt2_prompt": prompts[i],
                                "mt2_raw": raw_mt2[i],
                                "direct_prompt": None,
                                "direct_raw": None,
                            }
                        )
                else:
                    prompts = [direct_answer_prompt(q, target_lang) for q in x_l]
                    raw_direct = reasoner_engine.generate_batch(
                        prompts,
                        max_new_tokens=cfg.generation.mt1_max_new_tokens,
                        temperature=cfg.generation.mt1_temperature,
                        top_p=cfg.generation.mt1_top_p,
                    )
                    for i, ex_id in enumerate(example_ids):
                        out_rows_by_lang[lang].append(
                            {
                                "dataset": cfg.dataset.dataset_type,
                                "lang": lang,
                                "target_lang": target_lang,
                                "example_id": ex_id,
                                "error_group": "baseline",
                                "error_type": None,
                                "cascade_mode": mode,
                                "slice": f"baseline/{mode}",
                                "x_l": x_l[i],
                                "x_en": None,
                                "r_en": None,
                                "y_en": None,
                                "x_en_err": None,
                                "r_en_err": None,
                                "y_en_err": None,
                                "reasoner_prompt": None,
                                "reasoner_raw": None,
                                "mt2_prompt": None,
                                "mt2_raw": None,
                                "direct_prompt": prompts[i],
                                "direct_raw": raw_direct[i],
                            }
                        )

            if not baseline_only:
                for i, ex_id in enumerate(example_ids):
                    for v in input_map.get(ex_id, []):
                        err_type = v["error_type"]
                        x_err = v["x_en_err"]
                        if not x_err:
                            continue

                        reasoner_prompt_err = reasoner_prompt(x_err)
                        reasoner_raw_err = reasoner_engine.generate_batch(
                            [reasoner_prompt_err],
                            max_new_tokens=cfg.generation.mt1_max_new_tokens,
                            temperature=cfg.generation.mt1_temperature,
                            top_p=cfg.generation.mt1_top_p,
                        )[0]
                        parsed = _parse_reasoner(reasoner_raw_err, reasoner_prompt_err, x_err)
                        r_err = parsed["reasoning"]
                        y_err = parsed["answer"]

                        for mode in mode_list:
                            if mode == "direct":
                                continue
                            mt2_prompt = _mt2_prompt_for_mode(
                                mode,
                                x_l=x_l[i],
                                x_en=x_err,
                                r_en=r_err,
                                y_en=y_err,
                                target_lang=target_lang,
                            )
                            mt2_raw = mt2_engine.generate_batch(
                                [mt2_prompt],
                                max_new_tokens=cfg.generation.mt2_max_new_tokens,
                                temperature=cfg.generation.mt2_temperature,
                                top_p=cfg.generation.mt2_top_p,
                            )[0]
                            out_rows_by_lang[lang].append(
                                {
                                    "dataset": cfg.dataset.dataset_type,
                                    "lang": lang,
                                    "target_lang": target_lang,
                                    "example_id": ex_id,
                                    "error_group": "input_err",
                                    "error_type": err_type,
                                    "cascade_mode": mode,
                                    "slice": f"input_err/{err_type}/{mode}",
                                    "x_l": x_l[i],
                                    "x_en": x_err,
                                    "r_en": r_err,
                                    "y_en": y_err,
                                    "x_en_err": x_err,
                                    "r_en_err": None,
                                    "y_en_err": None,
                                    "reasoner_prompt": reasoner_prompt_err,
                                    "reasoner_raw": reasoner_raw_err,
                                    "mt2_prompt": mt2_prompt,
                                    "mt2_raw": mt2_raw,
                                    "direct_prompt": None,
                                    "direct_raw": None,
                                }
                            )
                LOGGER.info(
                    "lang=%s input error raw predictions done (non-direct modes=%s)",
                    lang,
                    [m for m in mode_list if m != "direct"],
                )

                for i, ex_id in enumerate(example_ids):
                    for v in output_map.get(ex_id, []):
                        err_type = v["error_type"]
                        r_err = v.get("r_en_err") or ""
                        y_err = v.get("y_en_err")
                        if not y_err:
                            continue
                        for mode in mode_list:
                            if mode == "direct":
                                continue
                            mt2_prompt = _mt2_prompt_for_mode(
                                mode,
                                x_l=x_l[i],
                                x_en=x_en[i],
                                r_en=r_err,
                                y_en=y_err,
                                target_lang=target_lang,
                            )
                            mt2_raw = mt2_engine.generate_batch(
                                [mt2_prompt],
                                max_new_tokens=cfg.generation.mt2_max_new_tokens,
                                temperature=cfg.generation.mt2_temperature,
                                top_p=cfg.generation.mt2_top_p,
                            )[0]
                            out_rows_by_lang[lang].append(
                                {
                                    "dataset": cfg.dataset.dataset_type,
                                    "lang": lang,
                                    "target_lang": target_lang,
                                    "example_id": ex_id,
                                    "error_group": "output_err",
                                    "error_type": err_type,
                                    "cascade_mode": mode,
                                    "slice": f"output_err/{err_type}/{mode}",
                                    "x_l": x_l[i],
                                    "x_en": x_en[i],
                                    "r_en": r_err,
                                    "y_en": y_err,
                                    "x_en_err": None,
                                    "r_en_err": r_err,
                                    "y_en_err": y_err,
                                    "reasoner_prompt": None,
                                    "reasoner_raw": None,
                                    "mt2_prompt": mt2_prompt,
                                    "mt2_raw": mt2_raw,
                                    "direct_prompt": None,
                                    "direct_raw": None,
                                }
                            )
                LOGGER.info(
                    "lang=%s output error raw predictions done (non-direct modes=%s)",
                    lang,
                    [m for m in mode_list if m != "direct"],
                )
            lang_out_path = os.path.join(out_root, f"{lang}.jsonl")
            _write_jsonl(lang_out_path, out_rows_by_lang[lang])
            LOGGER.info("lang=%s done raw_rows_written=%d", lang, len(out_rows_by_lang[lang]))
            LOGGER.info("lang=%s flushed raw predictions to %s", lang, lang_out_path)

        for lang, lang_rows in out_rows_by_lang.items():
            _write_jsonl(os.path.join(out_root, f"{lang}.jsonl"), lang_rows)
        LOGGER.info("run_raw_predictions done: wrote per-language raw predictions under %s", out_root)
        return out_root
    finally:
        for eng in engine_cache.values():
            try:
                eng.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Run v2 raw cascade predictions")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corruptions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--modes", default="")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()
    run_raw_predictions(
        config_path=args.config,
        corruption_jsonl=args.corruptions,
        out_jsonl=args.out,
        modes=args.modes,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()
