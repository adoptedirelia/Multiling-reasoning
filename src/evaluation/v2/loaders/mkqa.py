import json
from gzip import GzipFile
from typing import Dict, List

from .common import mk_record


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


def load_for_lang(mkqa_path: str, lang: str, max_examples: int) -> List[Dict]:
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
                q_l = (queries[lang] or "").strip()
                a_l_list = _extract_gold_strings(answers[lang])
                if not q_l or not a_l_list:
                    continue
                records.append(
                    mk_record(
                        dataset="mkqa",
                        lang=lang,
                        example_id=str(ex.get("example_id", ex.get("id", ""))),
                        x_l=q_l,
                        y_l_gold=a_l_list,
                    )
                )
                if len(records) >= max_examples:
                    break
    return records

