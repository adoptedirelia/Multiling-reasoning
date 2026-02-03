import json
from typing import Dict, List, Optional, Set

from datasets import load_dataset


def _extract_gold_strings(answer_list) -> List[str]:
    out: Set[str] = set()
    if not answer_list:
        return []

    if isinstance(answer_list, dict):
        answer_list = [answer_list]
    if not isinstance(answer_list, list):
        answer_list = [answer_list]

    for a in answer_list:
        if isinstance(a, dict):
            txt = (a.get("text") or "").strip()
            ent = (a.get("entity") or "").strip()
            if txt:
                out.add(txt)
            if ent:
                out.add(ent)

            aliases = a.get("aliases", [])
            if isinstance(aliases, list):
                for al in aliases:
                    s = str(al).strip()
                    if s:
                        out.add(s)
        else:
            s = str(a).strip()
            if s:
                out.add(s)

    return sorted(out)


def load_mkqa(lang: str, split: str = "train", max_examples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    ds = load_dataset("apple/mkqa", split=split, trust_remote_code=True).shuffle(seed=seed)
    out: List[Dict] = []
    for ex in ds:
        queries = ex.get("queries", {})
        answers = ex.get("answers", {})
        if "en" not in queries or "en" not in answers:
            continue
        if lang not in queries or lang not in answers:
            continue

        q_en = (queries["en"] or "").strip()
        q_l = (queries[lang] or "").strip()
        a_en_list = _extract_gold_strings(answers["en"])
        a_l_list = _extract_gold_strings(answers[lang])
        if not q_en or not q_l or not a_en_list or not a_l_list:
            continue

        out.append(
            {
                "id": str(ex.get("example_id", ex.get("id", ""))),
                "q_L": q_l,
                "q_en": q_en,
                "a_L_list": a_l_list,
                "a_en_list": a_en_list,
            }
        )
        if max_examples is not None and len(out) >= max_examples:
            break
    return out


def load_json_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list data in {path}")
    return data
