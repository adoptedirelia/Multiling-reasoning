from typing import Dict, List

from ..config import DatasetConfig
from . import aya, blend, global_piqa, mkqa


def load_records_for_language(dataset_cfg: DatasetConfig, lang: str) -> List[Dict]:
    if dataset_cfg.dataset_type == "mkqa":
        return mkqa.load_for_lang(dataset_cfg.mkqa_path, lang, dataset_cfg.max_examples)
    if dataset_cfg.dataset_type == "global_piqa":
        return global_piqa.load_for_lang(
            hf_name=dataset_cfg.hf_name or "",
            split=dataset_cfg.hf_split,
            max_examples=dataset_cfg.max_examples,
            lang=lang,
            hf_configs=dataset_cfg.hf_configs,
        )
    if dataset_cfg.dataset_type == "aya":
        return aya.load_for_lang(
            hf_name=dataset_cfg.hf_name or "",
            split=dataset_cfg.hf_split,
            max_examples=dataset_cfg.max_examples,
            lang=lang,
            hf_configs=dataset_cfg.hf_configs,
        )
    if dataset_cfg.dataset_type == "blend":
        return blend.load_for_lang(
            hf_name=dataset_cfg.hf_name or "",
            split=dataset_cfg.hf_split,
            max_examples=dataset_cfg.max_examples,
            lang=lang,
            hf_configs=dataset_cfg.hf_configs,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_cfg.dataset_type}")


def load_records_by_language(dataset_cfg: DatasetConfig) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for lang in dataset_cfg.langs:
        out[lang] = load_records_for_language(dataset_cfg, lang)
    return out
