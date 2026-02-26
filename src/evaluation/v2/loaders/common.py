from typing import Dict, List, Optional


def load_hf_split(name: str, cfg_name: Optional[str], split_name: str):
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


def normalize_targets(value) -> List[str]:
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


def mk_record(dataset: str, lang: str, example_id: str, x_l: str, y_l_gold: List[str], meta: Optional[Dict] = None):
    return {
        "dataset": dataset,
        "lang": lang,
        "example_id": str(example_id),
        "x_l": x_l,
        "y_l_gold": y_l_gold,
        "meta": meta or {},
    }

