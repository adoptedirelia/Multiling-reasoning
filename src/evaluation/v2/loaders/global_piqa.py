from typing import Dict, List, Optional

from .common import load_hf_split, mk_record

_GLOBAL_PIQA_CONFIG_ALIASES = {
    "arabic": "arb_arab",
    "japanese": "jpn_jpan",
    "vietnamese": "vie_latn",
    "chinese": "cmn_hans",
    "telugu": "tel_telu",
    "marathi": "mar_deva",
    "amharic": "amh_ethi",
}


def load_for_lang(
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

    ds = load_hf_split(hf_name, cfg_name, split)
    records: List[Dict] = []
    for idx, row in enumerate(ds):
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
        elif label == 1:
            ans = sol1
        else:
            ans = ""
        if not prompt or not ans:
            continue
        records.append(
            mk_record(
                dataset="global_piqa",
                lang=lang,
                example_id=str(row.get("example_id", idx)),
                x_l=prompt,
                y_l_gold=[ans],
            )
        )
        if len(records) >= max_examples:
            break
    return records

