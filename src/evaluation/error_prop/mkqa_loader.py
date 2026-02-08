import json
from gzip import GzipFile
from typing import Dict, List, Tuple


def _extract_gold_strings(answer_list) -> List[str]:
    out = set()
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


def load_mkqa_records(mkqa_path: str, lang: str, max_examples: int) -> List[Dict]:
    records: List[Dict] = []
    with open(mkqa_path, "rb") as f:
        with GzipFile(fileobj=f) as gz:
            for line in gz:
                ex = json.loads(line)
                queries = ex.get("queries", {})
                answers = ex.get("answers", {})
                if "en" not in queries or lang not in queries:
                    continue
                if "en" not in answers or lang not in answers:
                    continue
                q_en = (queries["en"] or "").strip()
                q_l = (queries[lang] or "").strip()
                a_l_list = _extract_gold_strings(answers[lang])
                a_en_list = _extract_gold_strings(answers.get("en", []))
                if not q_en or not q_l or not a_l_list:
                    continue
                records.append(
                    {
                        "example_id": str(ex.get("example_id", ex.get("id", ""))),
                        "x_en": q_en,
                        "x_l": q_l,
                        "y_l_gold": a_l_list,
                        "y_en_gold": a_en_list,
                    }
                )
                if len(records) >= max_examples:
                    break
    return records


def load_mkqa_records_for_ids(mkqa_path: str, lang: str, example_ids: List[str]) -> Dict[str, Dict]:
    ids = set(example_ids)
    records: Dict[str, Dict] = {}
    with open(mkqa_path, "rb") as f:
        with GzipFile(fileobj=f) as gz:
            for line in gz:
                ex = json.loads(line)
                ex_id = str(ex.get("example_id", ex.get("id", "")))
                if ex_id not in ids:
                    continue
                queries = ex.get("queries", {})
                answers = ex.get("answers", {})
                if "en" not in queries or lang not in queries:
                    continue
                if "en" not in answers or lang not in answers:
                    continue
                q_en = (queries["en"] or "").strip()
                q_l = (queries[lang] or "").strip()
                a_l_list = _extract_gold_strings(answers[lang])
                if not q_en or not q_l or not a_l_list:
                    continue
                records[ex_id] = {
                    "example_id": ex_id,
                    "x_en": q_en,
                    "x_l": q_l,
                    "y_l_gold": a_l_list,
                }
                if len(records) >= len(ids):
                    break
    return records


def write_mkqa_subset(mkqa_path: str, example_ids: List[str], out_path: str) -> None:
    ids = set(example_ids)
    with open(mkqa_path, "rb") as f_in:
        with GzipFile(fileobj=f_in) as gz_in:
            with open(out_path, "wb") as f_out:
                with GzipFile(fileobj=f_out, mode="wb") as gz_out:
                    for line in gz_in:
                        try:
                            ex = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ex_id = str(ex.get("example_id", ex.get("id", "")))
                        if ex_id in ids:
                            gz_out.write(line)
