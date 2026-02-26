import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

from .config import ErrorSimConfig, load_config
from .corruption_builder import build_corruptions
from .corruption_data import load_corruptions_jsonl
from .dataset_loader import load_records_by_language
from .engine_factory import create_engine
from .eval_utils import run_mkqa_eval, run_simple_eval, write_predictions_jsonl
from .mkqa_loader import write_mkqa_subset
from .models import ENReasoner, MT2Context, MT2Standard


def _engine_key(cfg):
    try:
        engine_kwargs_key = json.dumps(cfg.engine_kwargs, sort_keys=True, default=str)
    except TypeError:
        # Last-resort stability for unusual, non-JSON-serializable kwargs.
        engine_kwargs_key = repr(cfg.engine_kwargs)
    return (
        cfg.model_type,
        cfg.model_name,
        cfg.device_map,
        cfg.torch_dtype,
        cfg.attn_implementation,
        engine_kwargs_key,
    )


def run(config_path: str):
    cfg: ErrorSimConfig = load_config(config_path)
    log_dir = os.path.join(cfg.outputs.logs_dir, "error_sim")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=log_file,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    if not cfg.corruption.input_jsonl:
        cfg.corruption.input_jsonl = build_corruptions(config_path)

    rows = load_corruptions_jsonl(cfg.corruption.input_jsonl)
    rows_by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        lang = r.get("lang")
        if not lang:
            continue
        rows_by_lang[lang].append(r)

    engine_cache = {}

    def get_engine(mcfg):
        key = _engine_key(mcfg)
        if key in engine_cache:
            return engine_cache[key]
        eng = create_engine(mcfg)
        eng.load_model()
        engine_cache[key] = eng
        return eng

    reasoner = ENReasoner(get_engine(cfg.models["reasoner_en"]))
    mt2_engine = get_engine(cfg.models["mt2"])

    mkqa_dir = os.path.join(os.path.dirname(__file__), "ml-mkqa")
    records_by_lang = load_records_by_language(cfg.dataset)

    run_log_dir = os.path.join(cfg.outputs.logs_dir, "error_sim", cfg.run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    run_pred_intermediate_dir = os.path.join(
        cfg.outputs.predictions_dir, "_intermediate", cfg.run_name
    )
    os.makedirs(run_pred_intermediate_dir, exist_ok=True)

    for lang in cfg.dataset.langs:
        logging.info("Language=%s stage=load", lang)
        rec_list = records_by_lang.get(lang, [])
        if not rec_list:
            logging.warning("Language=%s has no dataset records; skipping", lang)
            continue
        rec_map = {r["example_id"]: r for r in rec_list}

        lang_rows = rows_by_lang.get(lang, [])
        if not lang_rows:
            logging.warning("Language=%s has no corruption rows; skipping", lang)
            continue

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
        if not example_ids:
            logging.warning("Language=%s has no overlap between rows and dataset; skipping", lang)
            continue

        if len(example_ids) > cfg.dataset.max_examples:
            example_ids = example_ids[: cfg.dataset.max_examples]
        if not example_ids:
            logging.warning("Language=%s has no examples after truncation; skipping", lang)
            continue

        x_l = [rec_map[ex_id]["x_l"] for ex_id in example_ids]
        y_l_gold = [rec_map[ex_id]["y_l_gold"] for ex_id in example_ids]
        x_en = [x_en_map[ex_id].strip() for ex_id in example_ids]

        logging.info("Language=%s stage=baseline_reasoner", lang)
        base = reasoner.run_batch(
            x_en,
            max_new_tokens=cfg.generation.mt1_max_new_tokens,
            temperature=cfg.generation.mt1_temperature,
            top_p=cfg.generation.mt1_top_p,
        )
        r_en_base = [b["reasoning"] for b in base]
        y_en_base = [b["answer"] for b in base]

        mt2_std = MT2Standard(mt2_engine, lang)
        mt2_ctx = MT2Context(mt2_engine, lang)

        lang_log_path = os.path.join(run_log_dir, f"{lang}.jsonl")
        log_f = open(lang_log_path, "w", encoding="utf-8", buffering=1)
        lang_intermediate_path = os.path.join(run_pred_intermediate_dir, f"{lang}.jsonl")
        inter_f = open(lang_intermediate_path, "w", encoding="utf-8", buffering=1)

        def log_row(row: Dict):
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            err_group = row.get("error_group")
            x_en_used = row.get("x_en")
            r_en_used = row.get("r_en")
            y_en_used = row.get("y_en")
            if err_group == "input_err" and row.get("x_en_err"):
                x_en_used = row.get("x_en_err")
            if err_group == "output_err":
                r_en_used = row.get("r_en_err") or row.get("r_en")
                y_en_used = row.get("y_en_err") or row.get("y_en")

            inter_row = {
                "run_name": row.get("run_name"),
                "lang": row.get("lang"),
                "example_id": row.get("example_id"),
                "error_group": err_group,
                "error_type": row.get("error_type"),
                "cascade": row.get("cascade"),
                "x_l": row.get("x_l"),
                "x_en": x_en_used,
                "r_en": r_en_used,
                "y_en": y_en_used,
                "y_l": row.get("y_l_pred"),
            }
            inter_f.write(json.dumps(inter_row, ensure_ascii=False) + "\n")
            inter_f.flush()

        def run_cascade(
            x_l_hat: List[str],
            x_en_hat: List[str],
            r_en_hat: List[str],
            y_en_hat: List[str],
            ids: List[str],
        ):
            std = mt2_std.run_batch(
                y_en_hat,
                max_new_tokens=cfg.generation.mt2_max_new_tokens,
                temperature=cfg.generation.mt2_temperature,
                top_p=cfg.generation.mt2_top_p,
            )
            ctx = mt2_ctx.run_batch(
                x_l_hat,
                x_en_hat,
                r_en_hat,
                y_en_hat,
                max_new_tokens=cfg.generation.mt2_max_new_tokens,
                temperature=cfg.generation.mt2_temperature,
                top_p=cfg.generation.mt2_top_p,
            )
            return {
                "standard": dict(zip(ids, std)),
                "context": dict(zip(ids, ctx)),
            }, std, ctx

        logging.info("Language=%s stage=baseline_mt2", lang)
        baseline_preds, baseline_std, baseline_ctx = run_cascade(
            x_l, x_en, r_en_base, y_en_base, example_ids
        )
        for i, ex_id in enumerate(example_ids):
            common = {
                "run_name": cfg.run_name,
                "lang": lang,
                "example_id": ex_id,
                "error_group": "baseline",
                "error_type": None,
                "x_en": x_en[i],
                "x_en_err": None,
                "r_en": r_en_base[i],
                "y_en": y_en_base[i],
                "y_en_err": None,
                "x_l": x_l[i],
                "y_l_gold": y_l_gold[i],
            }
            log_row({**common, "cascade": "standard", "y_l_pred": baseline_std[i]})
            log_row({**common, "cascade": "context", "y_l_pred": baseline_ctx[i]})

        logging.info("Language=%s stage=input_errors_mt2", lang)
        input_err_preds = defaultdict(dict)
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
                preds, std, ctx = run_cascade([x_l[i]], [x_err], [r_err], [y_err], [ex_id])
                for k, pred_map in preds.items():
                    input_err_preds[(err_type, k)][ex_id] = pred_map[ex_id]
                common = {
                    "run_name": cfg.run_name,
                    "lang": lang,
                    "example_id": ex_id,
                    "error_group": "input_err",
                    "error_type": err_type,
                    "x_en": x_en[i],
                    "x_en_err": x_err,
                    "r_en": r_err,
                    "y_en": y_err,
                    "y_en_err": None,
                    "x_l": x_l[i],
                    "y_l_gold": y_l_gold[i],
                }
                log_row({**common, "cascade": "standard", "y_l_pred": std[0]})
                log_row({**common, "cascade": "context", "y_l_pred": ctx[0]})

        logging.info("Language=%s stage=output_errors_mt2", lang)
        output_err_preds = defaultdict(dict)
        for i, ex_id in enumerate(example_ids):
            for v in output_map.get(ex_id, []):
                err_type = v["error_type"]
                y_err = v.get("y_en_err")
                r_err = v.get("r_en_err")
                if not y_err:
                    continue
                preds, std, ctx = run_cascade([x_l[i]], [x_en[i]], [r_err or ""], [y_err], [ex_id])
                for k, pred_map in preds.items():
                    output_err_preds[(err_type, k)][ex_id] = pred_map[ex_id]
                common = {
                    "run_name": cfg.run_name,
                    "lang": lang,
                    "example_id": ex_id,
                    "error_group": "output_err",
                    "error_type": err_type,
                    "x_en": x_en[i],
                    "x_en_err": None,
                    "r_en": r_err or "",
                    "y_en": y_en_base[i],
                    "y_en_err": y_err,
                    "x_l": x_l[i],
                    "y_l_gold": y_l_gold[i],
                }
                log_row({**common, "cascade": "standard", "y_l_pred": std[0]})
                log_row({**common, "cascade": "context", "y_l_pred": ctx[0]})

        def dump_and_eval(tag: str, preds: Dict[str, str]):
            # Keep prediction files aligned with the exact evaluation subset.
            # If some corruption path skipped an example, backfill with empty output.
            full_preds = {ex_id: preds.get(ex_id, "") for ex_id in example_ids}
            pred_path = os.path.join(cfg.outputs.predictions_dir, tag, f"{lang}.jsonl")
            write_predictions_jsonl(pred_path, full_preds)
            out_dir = os.path.join(cfg.outputs.metrics_dir, tag, lang)
            if cfg.dataset.dataset_type == "mkqa":
                subset_ann_path = os.path.join(run_log_dir, f"mkqa_subset_{lang}.jsonl.gz")
                write_mkqa_subset(cfg.dataset.mkqa_path, example_ids, subset_ann_path)
                run_mkqa_eval(
                    mkqa_dir,
                    subset_ann_path,
                    pred_path,
                    lang,
                    out_dir,
                )
            else:
                gold_answers = {ex_id: rec_map[ex_id]["y_l_gold"] for ex_id in example_ids}
                run_simple_eval(full_preds, gold_answers, lang, out_dir)

        logging.info("Language=%s stage=eval_baseline", lang)
        for k, preds in baseline_preds.items():
            dump_and_eval(f"baseline/{k}", preds)

        logging.info("Language=%s stage=eval_input_errors", lang)
        for (err_type, k), preds in input_err_preds.items():
            dump_and_eval(f"input_err/{err_type}/{k}", preds)

        logging.info("Language=%s stage=eval_output_errors", lang)
        for (err_type, k), preds in output_err_preds.items():
            dump_and_eval(f"output_err/{err_type}/{k}", preds)

        logging.info("Language=%s stage=done", lang)
        inter_f.close()
        log_f.close()

    for eng in engine_cache.values():
        try:
            eng.shutdown()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="error simulation")
    ap.add_argument("--config", required=True, help="Path to error simulation config JSON")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
