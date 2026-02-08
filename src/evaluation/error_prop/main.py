import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .config import ErrorSimConfig, load_config
from .engine_factory import create_engine
from .corruption_data import load_corruptions_jsonl
from .corruption_builder import build_corruptions_mkqa
from .eval_utils import run_mkqa_eval, write_predictions_jsonl
from .mkqa_loader import load_mkqa_records_for_ids, write_mkqa_subset
from .models import ENReasoner, MT2Context, MT2Standard


def _engine_key(cfg):
    return (
        cfg.model_type,
        cfg.model_name,
        cfg.device_map,
        cfg.torch_dtype,
        cfg.attn_implementation,
        tuple(sorted(cfg.engine_kwargs.items())),
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

    mt2_engine = get_engine(cfg.models["mt2"])

    mkqa_dir = os.path.join(os.path.dirname(__file__), "ml-mkqa")
    annotation_file = os.path.join(mkqa_dir, "dataset", "mkqa.jsonl.gz")

    rng = random.Random(cfg.dataset.seed)

    if not cfg.corruption.input_jsonl:
        cfg.corruption.input_jsonl = build_corruptions_mkqa(config_path)

    rows = load_corruptions_jsonl(cfg.corruption.input_jsonl)
    example_ids = sorted({r["example_id"] for r in rows})
    x_en_map = {}
    for r in rows:
        x_en_map[r["example_id"]] = r["x_en"]
    x_en = [x_en_map[ex_id] for ex_id in example_ids]

    # baseline reasoning on gold x_en for evaluation
    reasoner_engine = get_engine(cfg.models["reasoner_en"])
    reasoner = ENReasoner(reasoner_engine)
    base = reasoner.run_batch(
        x_en,
        max_new_tokens=cfg.generation.mt1_max_new_tokens,
        temperature=cfg.generation.mt1_temperature,
        top_p=cfg.generation.mt1_top_p,
    )
    r_en_base = [b["reasoning"] for b in base]
    y_en_base = [b["answer"] for b in base]

    input_map = {ex_id: [] for ex_id in example_ids}
    output_map = {ex_id: [] for ex_id in example_ids}
    for r in rows:
        ex_id = r["example_id"]
        grp = r.get("error_group", "")
        et = r.get("error_type", "")
        if grp == "input_err":
            input_map[ex_id].append(
                {
                    "error_type": et,
                    "x_en_err": r.get("x_en_err"),
                }
            )
        elif grp == "output_err":
            output_map[ex_id].append(
                {
                    "error_type": et,
                    "r_en_err": r.get("r_en_err"),
                    "y_en_err": r.get("y_en_err"),
                }
            )

    input_err_variants = [input_map[ex_id] for ex_id in example_ids]
    output_err_variants = [output_map[ex_id] for ex_id in example_ids]

    lang_records = {}
    lang_id_sets = []
    for lang in cfg.dataset.langs:
        rec_map = load_mkqa_records_for_ids(cfg.dataset.mkqa_path, lang, example_ids)
        lang_records[lang] = rec_map
        lang_id_sets.append(set(rec_map.keys()))
        if len(rec_map) < len(example_ids):
            logging.warning(
                "Language %s has %d/%d base examples; will take intersection",
                lang,
                len(rec_map),
                len(example_ids),
            )

    intersection_ids = set(example_ids)
    for s in lang_id_sets:
        intersection_ids &= s
    if not intersection_ids:
        raise ValueError("No common examples across selected languages.")

    if len(intersection_ids) < len(example_ids):
        keep_ids = [ex_id for ex_id in example_ids if ex_id in intersection_ids]
        idx_map = {ex_id: i for i, ex_id in enumerate(example_ids)}
        keep_idxs = [idx_map[ex_id] for ex_id in keep_ids]
        example_ids = keep_ids
        x_en = [x_en[i] for i in keep_idxs]
        r_en_base = [r_en_base[i] for i in keep_idxs]
        y_en_base = [y_en_base[i] for i in keep_idxs]
        input_err_variants = [input_err_variants[i] for i in keep_idxs]
        output_err_variants = [output_err_variants[i] for i in keep_idxs]

    for lang in cfg.dataset.langs:
        logging.info("Language=%s", lang)
        rec_map = lang_records[lang]
        x_l = [rec_map[ex_id]["x_l"] for ex_id in example_ids]
        y_l_gold = [rec_map[ex_id]["y_l_gold"] for ex_id in example_ids]

        log_dir = os.path.join(cfg.outputs.logs_dir, "error_sim", cfg.run_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{lang}.jsonl")
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)

        def log_row(row):
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()

        subset_ann_path = os.path.join(log_dir, f"mkqa_subset_{lang}.jsonl.gz")
        write_mkqa_subset(cfg.dataset.mkqa_path, example_ids, subset_ann_path)

        logging.info("Language=%s stage=mt2_init", lang)
        # MT2 outputs
        mt2_std = MT2Standard(mt2_engine, lang)
        mt2_ctx = MT2Context(mt2_engine, lang)

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
            preds = {
                "standard": dict(zip(ids, std)),
                "context": dict(zip(ids, ctx)),
            }
            return preds, std, ctx

        logging.info("Language=%s stage=baseline_mt2", lang)
        # baseline
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
        # input errors (apply all error types per example)
        input_err_preds = defaultdict(dict)
        for i, ex_id in enumerate(example_ids):
            variants = input_err_variants[i]
            if not variants:
                continue
            for v in variants:
                err_type = v["error_type"]
                x_err = v["x_en_err"]
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
        # output errors (apply all error types per example)
        output_err_preds = defaultdict(dict)
        for i, ex_id in enumerate(example_ids):
            variants = output_err_variants[i]
            if not variants:
                continue
            for v in variants:
                err_type = v["error_type"]
                y_err = v["y_en_err"]
                r_err = v["r_en_err"]
                preds, std, ctx = run_cascade(
                    [x_l[i]], [x_en[i]], [r_err], [y_err], [ex_id]
                )
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
                    "r_en": r_err,
                    "y_en": y_en_base[i],
                    "y_en_err": y_err,
                    "x_l": x_l[i],
                    "y_l_gold": y_l_gold[i],
                }
                log_row({**common, "cascade": "standard", "y_l_pred": std[0]})
                log_row({**common, "cascade": "context", "y_l_pred": ctx[0]})

        # write predictions + eval
        def dump_and_eval(tag: str, preds: Dict[str, str]):
            pred_path = os.path.join(cfg.outputs.predictions_dir, tag, f"{lang}.jsonl")
            write_predictions_jsonl(pred_path, preds)
            out_dir = os.path.join(cfg.outputs.metrics_dir, tag, lang)
            run_mkqa_eval(mkqa_dir, subset_ann_path, pred_path, lang, out_dir)

        logging.info("Language=%s stage=eval_baseline", lang)
        # baseline
        for k, preds in baseline_preds.items():
            dump_and_eval(f"baseline/{k}", preds)

        logging.info("Language=%s stage=eval_input_errors", lang)
        for (err_type, k), preds in input_err_preds.items():
            dump_and_eval(f"input_err/{err_type}/{k}", preds)

        logging.info("Language=%s stage=eval_output_errors", lang)
        for (err_type, k), preds in output_err_preds.items():
            dump_and_eval(f"output_err/{err_type}/{k}", preds)
        logging.info("Language=%s stage=done", lang)
        log_f.close()


def main():
    ap = argparse.ArgumentParser(description="MKQA error simulation (rule-based corruptions)")
    ap.add_argument("--config", required=True, help="Path to error simulation config JSON")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
