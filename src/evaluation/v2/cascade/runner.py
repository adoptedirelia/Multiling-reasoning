import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List

from ..config import V2Config, load_config
from ..loaders.registry import load_records_by_language
from ..runtime.engine_factory import create_engine
from ..runtime.language_names import target_language_name
from ..runtime.models import ENReasoner, MT2Context, MT2Standard
from ..runtime.prompts import direct_answer_prompt

LOGGER = logging.getLogger(__name__)


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
        return "results/v2/predictions"
    if p.endswith(".jsonl"):
        # Backward-compatible: if a file path is given, use its stem as directory.
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
        if m in {"standard", "context", "direct"} and m not in seen:
            seen.add(m)
            keep.append(m)
    if not keep:
        raise ValueError("No valid cascade modes. Use standard/context/direct/both.")
    return keep


def _extract_answer_from_completion(raw: str, prompt: str) -> str:
    text = (raw or "")
    p = (prompt or "")
    if p and text.startswith(p):
        text = text[len(p) :].lstrip()
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    close_idx = text.lower().find("</answer>")
    if close_idx != -1:
        pre = text[:close_idx]
        open_idx = pre.lower().rfind("<answer>")
        if open_idx != -1:
            return pre[open_idx + len("<answer>") :].strip()
        return pre.strip()
    return text.strip()


def run_cascade_predictions(config_path: str, corruption_jsonl: str = "", out_jsonl: str = "", modes: str = "") -> str:
    cfg: V2Config = load_config(config_path)
    corr_path = corruption_jsonl or cfg.corruption.input_jsonl or cfg.corruption.output_jsonl or cfg.outputs.corruption_jsonl
    if not corr_path:
        raise ValueError("Missing corruption JSONL path.")
    if not os.path.exists(corr_path):
        raise FileNotFoundError(corr_path)
    out_root = _predictions_root(out_jsonl or cfg.outputs.predictions_jsonl)
    mode_list = _mode_list(modes.split(",") if modes else cfg.cascade.modes)

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("run_cascade_predictions start: run_name=%s dataset=%s", cfg.run_name, cfg.dataset.dataset_type)
    LOGGER.info("inputs: corruptions=%s out_root=%s modes=%s", corr_path, out_root, mode_list)
    rows = _load_jsonl(corr_path)
    LOGGER.info("loaded corruption rows=%d", len(rows))
    rows_by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        lang = r.get("lang")
        if lang:
            rows_by_lang[lang].append(r)

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
        reasoner = ENReasoner(get_engine(cfg.models["reasoner_en"]))
        mt2_engine = get_engine(cfg.models["mt2"])

        for lang in cfg.dataset.langs:
            rec_list = records_by_lang.get(lang, [])
            if not rec_list:
                LOGGER.info("lang=%s skipped: no dataset records", lang)
                continue
            rec_map = {r["example_id"]: r for r in rec_list}
            lang_rows = rows_by_lang.get(lang, [])
            if not lang_rows:
                LOGGER.info("lang=%s skipped: no corruption rows", lang)
                continue
            LOGGER.info("lang=%s start dataset_records=%d corruption_rows=%d", lang, len(rec_list), len(lang_rows))

            x_en_map = {}
            input_map: Dict[str, List[Dict]] = defaultdict(list)
            output_map: Dict[str, List[Dict]] = defaultdict(list)
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
            target_lang = target_language_name(lang)

            base = reasoner.run_batch(
                x_en,
                max_new_tokens=cfg.generation.mt1_max_new_tokens,
                temperature=cfg.generation.mt1_temperature,
                top_p=cfg.generation.mt1_top_p,
            )
            r_en_base = [b["reasoning"] for b in base]
            y_en_base = [b["answer"] for b in base]

            mt2_std = MT2Standard(mt2_engine, target_lang)
            mt2_ctx = MT2Context(mt2_engine, target_lang)

            def run_mode(
                mode: str,
                x_l_in: List[str],
                x_en_in: List[str],
                r_en_in: List[str],
                y_en_in: List[str],
            ) -> List[str]:
                if mode == "standard":
                    return mt2_std.run_batch(
                        y_en_in,
                        max_new_tokens=cfg.generation.mt2_max_new_tokens,
                        temperature=cfg.generation.mt2_temperature,
                        top_p=cfg.generation.mt2_top_p,
                    )
                if mode == "direct":
                    prompts = [direct_answer_prompt(q, target_lang) for q in x_l_in]
                    raw = get_engine(cfg.models["reasoner_en"]).generate_batch(
                        prompts,
                        max_new_tokens=cfg.generation.mt1_max_new_tokens,
                        temperature=cfg.generation.mt1_temperature,
                        top_p=cfg.generation.mt1_top_p,
                    )
                    return [_extract_answer_from_completion(r, prompts[i]) for i, r in enumerate(raw)]
                return mt2_ctx.run_batch(
                    x_l_in,
                    x_en_in,
                    r_en_in,
                    y_en_in,
                    max_new_tokens=cfg.generation.mt2_max_new_tokens,
                    temperature=cfg.generation.mt2_temperature,
                    top_p=cfg.generation.mt2_top_p,
                )

            for mode in mode_list:
                LOGGER.info("lang=%s baseline mode=%s n=%d", lang, mode, len(example_ids))
                pred_list = run_mode(mode, x_l, x_en, r_en_base, y_en_base)
                for i, ex_id in enumerate(example_ids):
                    out_rows_by_lang[lang].append(
                        {
                            "dataset": cfg.dataset.dataset_type,
                            "lang": lang,
                            "example_id": ex_id,
                            "error_group": "baseline",
                            "error_type": None,
                            "cascade_mode": mode,
                            "slice": f"baseline/{mode}",
                            "x_l": x_l[i],
                            "x_en": x_en[i],
                            "r_en": r_en_base[i],
                            "y_en": y_en_base[i],
                            "prediction": pred_list[i],
                            "x_en_err": None,
                            "r_en_err": None,
                            "y_en_err": None,
                        }
                    )

            for i, ex_id in enumerate(example_ids):
                for v in input_map.get(ex_id, []):
                    err_type = v["error_type"]
                    x_err = v["x_en_err"]
                    if not x_err:
                        continue
                    err = reasoner.run_batch(
                        [x_err],
                        max_new_tokens=cfg.generation.mt1_max_new_tokens,
                        temperature=cfg.generation.mt1_temperature,
                        top_p=cfg.generation.mt1_top_p,
                    )[0]
                    r_err = err["reasoning"]
                    y_err = err["answer"]
                    for mode in mode_list:
                        if mode == "direct":
                            continue
                        pred = run_mode(mode, [x_l[i]], [x_err], [r_err], [y_err])[0]
                        out_rows_by_lang[lang].append(
                            {
                                "dataset": cfg.dataset.dataset_type,
                                "lang": lang,
                                "example_id": ex_id,
                                "error_group": "input_err",
                                "error_type": err_type,
                                "cascade_mode": mode,
                                "slice": f"input_err/{err_type}/{mode}",
                                "x_l": x_l[i],
                                "x_en": x_err,
                                "r_en": r_err,
                                "y_en": y_err,
                                "prediction": pred,
                                "x_en_err": x_err,
                                "r_en_err": None,
                                "y_en_err": None,
                            }
                        )
            LOGGER.info(
                "lang=%s input error predictions done (non-direct modes=%s)",
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
                        pred = run_mode(mode, [x_l[i]], [x_en[i]], [r_err], [y_err])[0]
                        out_rows_by_lang[lang].append(
                            {
                                "dataset": cfg.dataset.dataset_type,
                                "lang": lang,
                                "example_id": ex_id,
                                "error_group": "output_err",
                                "error_type": err_type,
                                "cascade_mode": mode,
                                "slice": f"output_err/{err_type}/{mode}",
                                "x_l": x_l[i],
                                "x_en": x_en[i],
                                "r_en": r_err,
                                "y_en": y_err,
                                "prediction": pred,
                                "x_en_err": None,
                                "r_en_err": r_err,
                                "y_en_err": y_err,
                            }
                        )
            LOGGER.info(
                "lang=%s output error predictions done (non-direct modes=%s)",
                lang,
                [m for m in mode_list if m != "direct"],
            )
            LOGGER.info("lang=%s done rows_written=%d", lang, len(out_rows_by_lang[lang]))

        os.makedirs(out_root, exist_ok=True)
        for lang, lang_rows in out_rows_by_lang.items():
            _write_jsonl(os.path.join(out_root, f"{lang}.jsonl"), lang_rows)
        LOGGER.info("run_cascade_predictions done: wrote per-language predictions under %s", out_root)
        return out_root
    finally:
        for eng in engine_cache.values():
            try:
                eng.shutdown()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Run v2 cascade predictions")
    ap.add_argument("--config", required=True)
    ap.add_argument("--corruptions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--modes", default="")
    args = ap.parse_args()
    run_cascade_predictions(
        config_path=args.config,
        corruption_jsonl=args.corruptions,
        out_jsonl=args.out,
        modes=args.modes,
    )


if __name__ == "__main__":
    main()
