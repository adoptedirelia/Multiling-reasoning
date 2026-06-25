import argparse
import json
import logging
import os
import re
from typing import Dict, List

from ..runtime.text_utils import extract_answer, extract_reasoning

LOGGER = logging.getLogger(__name__)


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _predictions_root(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return "results/v2/predictions"
    if p.endswith(".jsonl"):
        return os.path.splitext(p)[0]
    return p


def _extract_answer_from_completion(raw: str, prompt: str) -> str:
    text = (raw or "")
    p = (prompt or "")
    if p and text.startswith(p):
        text = text[len(p) :].lstrip()
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        for match in m:
            cleaned = match.strip()
            if cleaned:
                return cleaned
        return m[0].strip()
    close_idx = text.lower().find("</answer>")
    if close_idx != -1:
        pre = text[:close_idx]
        open_idx = pre.lower().rfind("<answer>")
        if open_idx != -1:
            return pre[open_idx + len("<answer>") :].strip()
        return pre.strip()
    return text.strip()


def _extract_direct_prediction(raw: str, prompt: str, strategy: str) -> str:
    body = _strip_prompt_echo(raw, prompt)
    if strategy == "tolerant":
        return extract_answer(body)
    return _extract_answer_from_completion(raw, prompt)


def _strip_prompt_echo(raw: str, prompt: str) -> str:
    r = (raw or "")
    p = (prompt or "")
    if not p:
        return r
    if r.startswith(p):
        return r[len(p) :].lstrip()
    r_l = r.lstrip()
    if r_l.startswith(p):
        return r_l[len(p) :].lstrip()
    if p in r:
        return r.rsplit(p, 1)[-1].lstrip()
    return r


def _invalid_reasoner_pair(reasoning: str, answer: str) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    if "<think>" in lower or "<question>" in lower or "</answer>" in lower:
        return True
    if "```" in answer:
        return True
    return False


def _sanitize_answer_fallback(answer: str, question_en: str) -> str:
    if not answer:
        return question_en
    cleaned = answer.strip()
    if not cleaned:
        return question_en
    return cleaned


def _parse_reasoner(raw: str, prompt: str, question_en: str) -> Dict[str, str]:
    body = _strip_prompt_echo(raw, prompt)
    reasoning = extract_reasoning(body)
    answer = extract_answer(body)
    if _invalid_reasoner_pair(reasoning, answer):
        answer = _sanitize_answer_fallback(answer, question_en)
        reasoning = ""
    return {"reasoning": reasoning, "answer": answer}


def extract_predictions(raw_dir: str, out_jsonl: str = "", direct_strategy: str = "legacy") -> str:
    out_root = _predictions_root(out_jsonl)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(raw_dir)
    in_files = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".jsonl") and os.path.isfile(os.path.join(raw_dir, f))
    )
    if not in_files:
        raise ValueError(f"No .jsonl files found in {raw_dir}")

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("extract_predictions start: raw_dir=%s out_root=%s", raw_dir, out_root)

    os.makedirs(out_root, exist_ok=True)
    for file_name in in_files:
        in_path = os.path.join(raw_dir, file_name)
        rows = _load_jsonl(in_path)
        out_rows: List[Dict] = []
        for r in rows:
            mode = r.get("cascade_mode")
            if mode == "direct":
                prediction = _extract_direct_prediction(
                    r.get("direct_raw") or "",
                    r.get("direct_prompt") or "",
                    direct_strategy,
                )
                x_en = None
                r_en = None
                y_en = None
                out_rows.append(
                    {
                        "dataset": r.get("dataset"),
                        "lang": r.get("lang"),
                        "example_id": r.get("example_id"),
                        "error_group": r.get("error_group"),
                        "error_type": r.get("error_type"),
                        "cascade_mode": mode,
                        "slice": r.get("slice"),
                        "x_l": r.get("x_l"),
                        "x_en": x_en,
                        "r_en": r_en,
                        "y_en": y_en,
                        "prediction": prediction,
                        "x_en_err": r.get("x_en_err"),
                        "r_en_err": r.get("r_en_err"),
                        "y_en_err": r.get("y_en_err"),
                    }
                )
                continue
            else:
                prediction = extract_answer(_strip_prompt_echo(r.get("mt2_raw") or "", r.get("mt2_prompt") or ""))

            if r.get("error_group") == "output_err":
                x_en = r.get("x_en")
                r_en = r.get("r_en_err") or ""
                y_en = r.get("y_en_err")
            else:
                q_en = r.get("x_en")
                if r.get("error_group") == "input_err":
                    q_en = r.get("x_en_err") or q_en
                parsed = _parse_reasoner(r.get("reasoner_raw") or "", r.get("reasoner_prompt") or "", q_en or "")
                x_en = q_en
                r_en = parsed["reasoning"]
                y_en = parsed["answer"]

            out_rows.append(
                {
                    "dataset": r.get("dataset"),
                    "lang": r.get("lang"),
                    "example_id": r.get("example_id"),
                    "error_group": r.get("error_group"),
                    "error_type": r.get("error_type"),
                    "cascade_mode": mode,
                    "slice": r.get("slice"),
                    "x_l": r.get("x_l"),
                    "x_en": x_en,
                    "r_en": r_en,
                    "y_en": y_en,
                    "prediction": prediction,
                    "x_en_err": r.get("x_en_err"),
                    "r_en_err": r.get("r_en_err"),
                    "y_en_err": r.get("y_en_err"),
                }
            )

        out_path = os.path.join(out_root, file_name)
        _write_jsonl(out_path, out_rows)
        LOGGER.info("lang_file=%s extracted_rows=%d", file_name, len(out_rows))

    LOGGER.info("extract_predictions done: wrote extracted predictions under %s", out_root)
    return out_root


def main():
    ap = argparse.ArgumentParser(description="Extract current-format predictions from v2 raw predictions")
    ap.add_argument("--raw", required=True, help="Directory containing per-language raw jsonl files")
    ap.add_argument("--out", default="", help="Output directory for extracted per-language prediction files")
    ap.add_argument(
        "--direct-strategy",
        choices=("legacy", "tolerant"),
        default="legacy",
        help="How to extract direct/e2e predictions. 'legacy' preserves the original behavior; "
        "'tolerant' uses the shared answer extractor and strips leaked reasoning.",
    )
    args = ap.parse_args()
    extract_predictions(raw_dir=args.raw, out_jsonl=args.out, direct_strategy=args.direct_strategy)


if __name__ == "__main__":
    main()
