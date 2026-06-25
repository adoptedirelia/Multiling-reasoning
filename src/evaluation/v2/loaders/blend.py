import json
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .common import mk_record, normalize_targets


_LANG_TO_REGION = {
    "AS": "Assam",
    "AZ": "Azerbaijan",
    "CN": "China",
    "DZ": "Algeria",
    "ES": "Spain",
    "ET": "Ethiopia",
    "GB": "UK",
    "GR": "Greece",
    "ID": "Indonesia",
    "IR": "Iran",
    "JB": "West_Java",
    "KP": "North_Korea",
    "KR": "South_Korea",
    "MX": "Mexico",
    "NG": "Northern_Nigeria",
    "US": "US",
}


def _annotation_items(raw) -> Iterable[Dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        values = list(raw.values())
        if values and all(isinstance(v, list) for v in values):
            length = max(len(v) for v in values)
            items = []
            for idx in range(length):
                item = {}
                for key, vals in raw.items():
                    item[key] = vals[idx] if idx < len(vals) else None
                items.append(item)
            return items
        return [raw]
    return []


def _extend_unique(out: List[str], values) -> None:
    for value in normalize_targets(values):
        if value not in out:
            out.append(value)


def _load_annotations_by_id(hf_name: str, lang: str) -> Dict[str, Dict]:
    region = _LANG_TO_REGION.get(lang, lang)
    path = hf_hub_download(
        repo_id=hf_name,
        repo_type="dataset",
        filename=f"data/annotations_hf/{region}_data.json",
        local_files_only=True,
    )
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {
        str(row.get("ID")): row
        for row in rows
        if isinstance(row, dict) and row.get("ID") is not None
    }


def load_for_lang(
    hf_name: str,
    split: str,
    max_examples: int,
    lang: str,
    hf_configs: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    cfg_name = "short-answer-questions"
    if hf_configs:
        cfg_name = hf_configs.get(lang, cfg_name)
        if cfg_name is None:
            return []

    ds_by_lang = load_dataset(hf_name, cfg_name)
    if lang not in ds_by_lang:
        return []
    ds = ds_by_lang[lang]
    annotations_by_id = _load_annotations_by_id(hf_name, lang)

    records: List[Dict] = []
    for idx, row in enumerate(ds):
        x_l = (row.get("Question") or row.get("question") or "").strip()
        ex_id = row.get("ID")
        if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
            ex_id = row.get("id")
        if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
            ex_id = row.get("example_id")
        if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
            ex_id = f"{lang}_{idx}"

        ann_row = annotations_by_id.get(str(ex_id), row)
        y_l_gold: List[str] = []
        y_en_gold: List[str] = []
        for ann in _annotation_items(ann_row.get("annotations")):
            _extend_unique(y_l_gold, ann.get("answers"))
            _extend_unique(y_en_gold, ann.get("en_answers"))

        if not x_l or not y_l_gold:
            continue

        records.append(
            mk_record(
                dataset="blend",
                lang=lang,
                example_id=str(ex_id),
                x_l=x_l,
                y_l_gold=y_l_gold,
                meta={
                    "en_answers": y_en_gold,
                    "en_question": ann_row.get("en_question") or row.get("Translation"),
                },
            )
        )
        if len(records) >= max_examples:
            break
    return records
