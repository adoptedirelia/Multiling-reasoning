import argparse
import logging
from typing import Dict, List, Tuple

from .config import EvaluationConfig, ModelEngineConfig, load_config
from .data import load_json_examples, load_mkqa
from .engine_factory import create_engine
from .io_utils import append_row_csv, load_constraints, setup_logging
from .models import Corrector, LLMReasoner, LLMTranslator
from .pipeline import run_pipeline_constraints


def _engine_key(cfg: ModelEngineConfig) -> Tuple:
    return (
        cfg.model_type,
        cfg.model_name,
        cfg.device_map,
        cfg.torch_dtype,
        cfg.attn_implementation,
        tuple(sorted(cfg.engine_kwargs.items())),
    )


def _load_examples(config: EvaluationConfig, lang: str) -> List[Dict]:
    if config.dataset.name == "mkqa":
        return load_mkqa(
            lang=lang,
            split=config.dataset.split,
            max_examples=config.dataset.max_examples,
            seed=config.dataset.seed,
        )
    if config.dataset.name == "json":
        if not config.dataset.path:
            raise ValueError("dataset.path is required when dataset.name=json")
        return load_json_examples(config.dataset.path)
    raise ValueError(f"Unsupported dataset.name={config.dataset.name}")


def _build_fieldnames(constraints: List[Dict]) -> List[str]:
    base = ["dataset", "lang", "MT1_model", "reasoner_en_model", "MT2_model", "mt2_mode", "num_examples"]
    fields = []
    for c in constraints:
        c_id = c["id"]
        fields.extend(
            [
                f"constraint_{c_id}_acc_base",
                f"constraint_{c_id}_acc_err",
                f"constraint_{c_id}_acc_gap",
                f"constraint_{c_id}_base_ok_to_err_bad_rate",
                f"_sample_{c_id}_base_0",
                f"_sample_{c_id}_err_0",
            ]
        )
    return base + fields


def run(config_path: str):
    config = load_config(config_path)
    log_path = setup_logging(config.outputs.logs_dir, config.run_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Loaded config: %s", config_path)

    constraints = load_constraints(config.constraints_file)
    logging.info("Loaded %d constraints from %s", len(constraints), config.constraints_file)

    fieldnames = _build_fieldnames(constraints)
    engine_cache: Dict[Tuple, object] = {}

    def get_engine(model_cfg: ModelEngineConfig):
        key = _engine_key(model_cfg)
        if key in engine_cache:
            return engine_cache[key]
        engine = create_engine(model_cfg)
        engine.load_model()
        engine_cache[key] = engine
        return engine

    mt1_cfg = config.models["mt1"]
    reasoner_cfg = config.models["reasoner_en"]
    mt2_cfg = config.models["mt2"]

    mt1_engine = get_engine(mt1_cfg)
    reasoner_engine = get_engine(reasoner_cfg)
    mt2_engine = get_engine(mt2_cfg)

    for lang in config.dataset.langs:
        logging.info("Running language=%s", lang)
        examples = _load_examples(config, lang)
        if not examples:
            logging.warning("No examples for language=%s; skipping", lang)
            continue

        translator_l2en = LLMTranslator(
            mt1_engine,
            src_lang=lang,
            tgt_lang="en",
            default_max_new_tokens=config.generation.translation_max_new_tokens,
        )
        reasoner = LLMReasoner(
            reasoner_engine,
            default_max_new_tokens=config.generation.reasoning_max_new_tokens,
            default_temperature=config.generation.reasoning_temperature,
            default_top_p=config.generation.reasoning_top_p,
        )
        translator_en2l = LLMTranslator(
            mt2_engine,
            src_lang="en",
            tgt_lang=lang,
            default_max_new_tokens=config.generation.translation_max_new_tokens,
        )
        mt2_mode = (config.generation.mt2_mode or "translator").lower()
        if mt2_mode == "translator":
            mt2_component = translator_en2l
        elif mt2_mode == "corrector":
            mt2_component = Corrector(
                mt2_engine,
                target_lang=lang,
                default_max_new_tokens=config.generation.mt2_max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported generation.mt2_mode={config.generation.mt2_mode}")

        results = run_pipeline_constraints(
            dataset_name=config.dataset.name,
            examples=examples,
            translator_l2en=translator_l2en,
            translator_en2l=translator_en2l,
            reasoner=reasoner,
            mt2_component=mt2_component,
            lang_l=lang,
            constraints=constraints,
            max_examples=config.dataset.max_examples,
            translation_batch_size=config.generation.translation_batch_size,
            translation_max_new_tokens=config.generation.translation_max_new_tokens,
            reasoner_batch_size=config.generation.reasoning_batch_size,
            reasoner_max_new_tokens=config.generation.reasoning_max_new_tokens,
            reasoner_temperature=config.generation.reasoning_temperature,
            reasoner_top_p=config.generation.reasoning_top_p,
            mt2_batch_size=config.generation.mt2_batch_size,
            mt2_max_new_tokens=config.generation.mt2_max_new_tokens,
        )
        results["MT1_model"] = mt1_cfg.model_name
        results["reasoner_en_model"] = reasoner_cfg.model_name
        results["MT2_model"] = mt2_cfg.model_name
        results["mt2_mode"] = mt2_mode

        append_row_csv(config.outputs.results_csv, fieldnames, results)
        logging.info("Wrote results row for language=%s to %s", lang, config.outputs.results_csv)

    logging.info("Done")


def main():
    parser = argparse.ArgumentParser(description="LLM-only constraints evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to evaluation run config (JSON)")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
