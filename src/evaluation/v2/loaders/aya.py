from typing import Dict, List, Optional

from .common import load_hf_split, mk_record, normalize_targets


def load_for_lang(
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

    ds = load_hf_split(hf_name, cfg_name, split)
    records: List[Dict] = []
    # Aya Dolly-specific script filtering:
    # - Arabic must be language=arb with script=Arab
    # - Chinese must be language=zho with script=Hans
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
        x_l = (row.get("inputs") or "").strip()
        y_l_gold = normalize_targets(row.get("targets"))
        ex_id = row.get("id")
        if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
            ex_id = row.get("example_id")
        if ex_id is None or (isinstance(ex_id, str) and not ex_id.strip()):
            ex_id = f"{lang}_{idx}"
        if not x_l or not y_l_gold:
            continue
        records.append(
            mk_record(
                dataset="aya",
                lang=lang,
                example_id=str(ex_id),
                x_l=x_l,
                y_l_gold=y_l_gold,
            )
        )
        if len(records) >= max_examples:
            break
    return records
