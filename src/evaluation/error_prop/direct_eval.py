import argparse
import json
import logging
import os
from typing import Dict, List

from .dataset_loader import load_records_for_language
from .direct_config import DirectEvalConfig, load_direct_config
from .engine_factory import create_engine
from .eval_utils import run_mkqa_eval, run_simple_eval, write_predictions_jsonl
from .mkqa_loader import load_mkqa_records, write_mkqa_subset
from .prompts import direct_answer_prompt
from .text_utils import extract_answer


def run(config_path: str):
    cfg: DirectEvalConfig = load_direct_config(config_path)
    log_dir = os.path.join(cfg.outputs.logs_dir, "direct_eval")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=log_file,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    engine = create_engine(cfg.model)
    engine.load_model()

    mkqa_dir = os.path.join(os.path.dirname(__file__), "ml-mkqa")
    annotation_file = os.path.join(mkqa_dir, "dataset", "mkqa.jsonl.gz")

    for lang in cfg.dataset.langs:
        logging.info("Language=%s stage=load", lang)
        if cfg.dataset.dataset_type == "mkqa":
            records = load_mkqa_records(cfg.dataset.mkqa_path, lang, cfg.dataset.max_examples)
        else:
            records = load_records_for_language(cfg.dataset, lang)

        if not records:
            logging.warning("Language=%s has no records; skipping", lang)
            continue

        example_ids = [r["example_id"] for r in records]
        x_l = [r["x_l"] for r in records]
        y_l_gold = [r["y_l_gold"] for r in records]
        prompts = [direct_answer_prompt(q, lang) for q in x_l]

        logging.info("Language=%s stage=inference count=%d", lang, len(prompts))
        raw_outputs = engine.generate_batch(
            prompts,
            max_new_tokens=cfg.generation.mt2_max_new_tokens,
            temperature=cfg.generation.mt2_temperature,
            top_p=cfg.generation.mt2_top_p,
        )
        preds_list = [extract_answer(o) for o in raw_outputs]
        preds = dict(zip(example_ids, preds_list))

        pred_path = os.path.join(cfg.outputs.predictions_dir, cfg.run_name, f"{lang}.jsonl")
        write_predictions_jsonl(pred_path, preds)

        run_log_dir = os.path.join(cfg.outputs.logs_dir, "direct_eval", cfg.run_name)
        os.makedirs(run_log_dir, exist_ok=True)
        with open(os.path.join(run_log_dir, f"{lang}.jsonl"), "w", encoding="utf-8") as lf:
            for ex_id, q, gold, pred in zip(example_ids, x_l, y_l_gold, preds_list):
                lf.write(
                    json.dumps(
                        {
                            "run_name": cfg.run_name,
                            "lang": lang,
                            "example_id": ex_id,
                            "x_l": q,
                            "y_l_gold": gold,
                            "y_l_pred": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        out_dir = os.path.join(cfg.outputs.metrics_dir, cfg.run_name, lang)
        logging.info("Language=%s stage=eval", lang)
        if cfg.dataset.dataset_type == "mkqa":
            subset_ann_path = os.path.join(run_log_dir, f"mkqa_subset_{lang}.jsonl.gz")
            write_mkqa_subset(cfg.dataset.mkqa_path, example_ids, subset_ann_path)
            run_mkqa_eval(mkqa_dir, subset_ann_path, pred_path, lang, out_dir)
        else:
            gold_map: Dict[str, List[str]] = {
                r["example_id"]: r["y_l_gold"] for r in records
            }
            run_simple_eval(preds, gold_map, lang, out_dir)
        logging.info("Language=%s stage=done", lang)

    try:
        engine.shutdown()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Direct per-language evaluation (no cascade)")
    parser.add_argument("--config", required=True, help="Path to direct eval config JSON")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
