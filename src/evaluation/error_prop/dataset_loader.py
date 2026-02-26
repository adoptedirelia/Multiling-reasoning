from typing import Dict, List, Optional

from .config import DatasetConfig
from .mkqa_loader import load_mkqa_records


_GLOBAL_PIQA_CONFIG_ALIASES = {
    "arabic": "arb_arab",
    "japanese": "jpn_jpan",
    "vietnamese": "vie_latn",
    "chinese": "cmn_hans",
    "telugu": "tel_telu",
    "marathi": "mar_deva",
    "amharic": "amh_ethi",
}


def _row_to_global_piqa_example(row: Dict, idx: int, lang: str) -> Dict:
    prompt = (row.get("prompt") or "").strip()
    sol0 = (row.get("solution0") or "").strip()
    sol1 = (row.get("solution1") or "").strip()
    label_raw = row.get("label")
    try:
        label = int(label_raw)
    except (TypeError, ValueError):
        label = None

    if label == 0:
        ans = sol0
        y_en = row.get("gemini_translated0")
    elif label == 1:
        ans = sol1
        y_en = row.get("gemini_translated1")
    else:
        ans = ""
        y_en = None
    return {
        "example_id": str(row.get("example_id", idx)),
        "lang": lang,
        "x_l": prompt,
        # We always generate x_en through MT1 in the pipeline.
        "x_en": "",
        "y_l_gold": [ans] if ans else [],
        "y_en_gold": [y_en.strip()] if isinstance(y_en, str) and y_en.strip() else [],
    }


def _load_hf_split(name: str, cfg_name: Optional[str], split_name: str):
    from datasets import load_dataset

    try:
        if cfg_name is None:
            return load_dataset(name, split=split_name)
        return load_dataset(name, cfg_name, split=split_name)
    except ValueError as e:
        msg = str(e)
        if 'Unknown split "' in msg and split_name != "test":
            if cfg_name is None:
                return load_dataset(name, split="test")
            return load_dataset(name, cfg_name, split="test")
        raise


def _normalize_targets(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    return []


def _row_to_aya_example(row: Dict, idx: int, lang: str) -> Dict:
    x_l = (row.get("inputs") or "").strip()
    y_l_gold = _normalize_targets(row.get("targets"))
    ex_id = row.get("id")
    if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
        ex_id = row.get("example_id")
    if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
        ex_id = f"{lang}_{idx}"
    return {
        "example_id": str(ex_id),
        "lang": lang,
        "x_l": x_l,
        # We always generate x_en through MT1 in the pipeline.
        "x_en": "",
        "y_l_gold": y_l_gold,
        "y_en_gold": [],
    }


def load_global_piqa_records_for_lang(
    hf_name: str,
    split: str,
    max_examples: int,
    lang: str,
    hf_configs: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    cfg_name = None
    if hf_configs:
        raw = hf_configs.get(lang)
        if raw is None:
            return []
        cfg_name = _GLOBAL_PIQA_CONFIG_ALIASES.get(raw, raw)

    ds = _load_hf_split(hf_name, cfg_name, split)
    records: List[Dict] = []
    for idx, row in enumerate(ds):
        ex = _row_to_global_piqa_example(row, idx, lang)
        if ex.get("x_l") and ex.get("y_l_gold"):
            records.append(ex)
        if len(records) >= max_examples:
            break
    return records


def load_aya_records_for_lang(
    hf_name: str,
    split: str,
    max_examples: int,
    lang: str,
    hf_configs: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    cfg_name = None
    if hf_configs:
        cfg_name = hf_configs.get(lang)
        if cfg_name is None:
            return []

    ds = _load_hf_split(hf_name, cfg_name, split)
    records: List[Dict] = []
    # Aya Dolly-specific script filtering requested by experiment setup:
    # - Arabic: language=arb and script=Arab
    # - Chinese: language=zho and script=Hans
    required_script = None
    if cfg_name == "dolly_machine_translated":
        if lang.lower() == "arb":
            required_script = "arab"
        elif lang.lower() == "zho":
            required_script = "hans"
    for idx, row in enumerate(ds):
        row_lang = (row.get("language") or "").strip().lower()
        if row_lang != lang.lower():
            continue
        if required_script is not None:
            row_script = (row.get("script") or "").strip().lower()
            if row_script != required_script:
                continue
        ex = _row_to_aya_example(row, idx, lang)
        if ex.get("x_l") and ex.get("y_l_gold"):
            records.append(ex)
        if len(records) >= max_examples:
            break
    return records


def load_records_for_language(
    dataset_cfg: DatasetConfig, lang: str, max_examples_override: Optional[int] = None
) -> List[Dict]:
    max_examples = max_examples_override if max_examples_override is not None else dataset_cfg.max_examples
    if dataset_cfg.dataset_type == "mkqa":
        return load_mkqa_records(dataset_cfg.mkqa_path, lang, max_examples)
    if dataset_cfg.dataset_type == "global_piqa":
        if not dataset_cfg.hf_name:
            raise ValueError("dataset.hf_name is required for global_piqa")
        return load_global_piqa_records_for_lang(
            dataset_cfg.hf_name,
            dataset_cfg.hf_split,
            max_examples,
            lang,
            dataset_cfg.hf_configs,
        )
    if dataset_cfg.dataset_type == "aya":
        if not dataset_cfg.hf_name:
            raise ValueError("dataset.hf_name is required for aya")
        return load_aya_records_for_lang(
            dataset_cfg.hf_name,
            dataset_cfg.hf_split,
            max_examples,
            lang,
            dataset_cfg.hf_configs,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_cfg.dataset_type}")


def load_records_by_language(
    dataset_cfg: DatasetConfig, max_examples_override: Optional[int] = None
) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for lang in dataset_cfg.langs:
        out[lang] = load_records_for_language(dataset_cfg, lang, max_examples_override)
    return out
