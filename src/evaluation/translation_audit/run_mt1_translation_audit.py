#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from src.eval.engine.openai_ import OpenAIEngine


LOGGER = logging.getLogger(__name__)

DATASETS = ("aya", "blend", "global_piqa", "mkqa")
MODELS = ("llama", "mistral", "gpt")
EXPORT_MODEL_FILES = {
    "gpt": "gpt-30-samples",
    "llama": "llama",
    "mistral": "mistral",
}

PROMPT_TEMPLATE = """You are a translation quality auditor. You will be given a source-language text and a candidate translation into English. The input comes in two forms: some items are a question only; others bundle the answer options into the same sentence as the question. In both cases, compare the candidate translation against the source and decide whether it contains an error.
Error categories:
1. Structural / formatting error

The output is not a clean translation — it leaks annotation templates, repeats identical translations for distinct options, drops the question body, uses placeholders, outputs meta text, or leaves untranslated source-script text inside the English output.

Source = "正确翻译（ Cho tam giác ABCABC
ABC nội tiếp trên đường tròn ω\\omega
ω. ... Tính giá trị m+nm+n
m+n.  ）："

Correct: only the Vietnamese question, translated (no template wrapper).

Error: the stored translation keeps the Chinese annotation template 正确翻译（…）： wrapped around the text instead of containing only the translated question.

2. Referent / entity substitution

The overall topic changes because a key referent is swapped for a different one. This includes person, team, place, organization, object, instrument, food item, artifact, or other central noun phrase.

Source = "ラムズはいつスーパーボウルでプレーしましたか"

Correct: "When did the Rams play in the Super Bowl?"

Error: "When was the Super Bowl that the Lions played in?" — the team Rams is replaced by Lions.

Source = "He put the flute in a hot room."

Correct: "He put the flute in a hot room."

Error: "He put the clock in a hot room." — the main object changes from flute to clock.

3. Event / constraint distortion

The same core referents remain, but who-did-what, the key action/relation, negation, condition, comparison, or quantity is changed or dropped.

Source = "Akeredolu fofin de awọn ọlọkada l'Ondo."

Correct: "Akeredolu has given new motorcycles to the riders in Ondo."

Error: "Akeredolu appoints new Ondo commissioners" — the action "giving motorcycles" becomes "appointing commissioners."

Source = "Who was the bridge sold to?"

Correct: "Who was the bridge sold to?"

Error: "Who sold the bridge?" — the same bridge remains, but the relation changes.

4. Cultural / local-term mistranslation

A culture-specific food, idiom, institution, festival, clothing item, household item, or local artifact is translated literally or mapped to the wrong referent.

Source = "Um misto quente é um sanduíche feito com pão."

Correct: 'A "misto quente" is a sandwich made with bread.'

Error: "A hot mix is a sandwich made with bread" — the Brazilian food term "misto quente" (ham-and-cheese toastie) is literally rendered as "hot mix."

5. Hallucination / over-answering

The model invents content not in the source, replaces the source with an unrelated question or statement, or solves/explains the task instead of translating it.

Source (a math problem) = "A group of 7 friends split a bill of $84 equally. How much does each pay?"

Correct: preserves the question.

Error: "Each person pays $12." — the model answered the problem instead of translating it, producing a number absent from the source.

Source = "What snack is popular at a local fair?"

Correct: "What snack is popular at a local fair?"

Error: "What is the current role of the national parliament?" — this invents a different, unrelated question.

Decision order (when more than one category could apply, assign the FIRST that matches):
1 -> The output is not a clean translation: it leaks code / templates / annotations / placeholders, drops the question body, repeats or merges options, outputs meta text like "unknown question", OR leaves part of the text untranslated in the source script.
5 -> Otherwise, the model invents content absent from the source, replaces the source with an unrelated question or statement, or answers / solves / explains the task instead of translating it.
2 -> Otherwise, a key referent is swapped for a DIFFERENT one (person / team / place / org / object / instrument / item / topic). A mere spelling / transliteration variant of the SAME referent (e.g. "Ishikawa" -> "Ishekawa") is NOT category 2 — that is OK.
4 -> Otherwise, a culture-specific term (food / idiom / institution / festival / clothing / artifact) is mapped to the wrong referent.
3 -> Otherwise, the same core referents remain but an action / relation / quantity / condition / comparison / negation is changed or dropped. Use 3 ONLY as the residual meaning error — never as a catch-all for messy output (that is 1), unrelated invented content (that is 5), or referent swaps (that are 2).
OK -> none of the above; the translation is faithful.

Mark OK when the meaning is preserved even if the English wording changes. Do NOT mark an error just because:
- the translation is a paraphrase rather than literal
- a search-query fragment stays a fragment
- a bare name is copied as a bare name
- singular/plural, article choice, or phrasing differences do not change meaning
- a close wording variant like "type" vs "kind" preserves the same meaning

The source may be a full sentence, a question fragment, a title, a search query, or just a name. Judge meaning fidelity, not grammatical style.

Output format:

Return a single JSON object and nothing else (no preamble, no markdown fences):
json{{
  "error type": ""
}}
Rules:

1. question: the source-language text, copied verbatim.
2. translation: the candidate English translation, copied verbatim.
3. error type: one of 1, 2, 3, 4, 5, or OK if the translation is faithful.
4. For items with bundled options, a single dropped or merged option still counts as category 1.
5. A correct answer value does not make the translation correct; judge by meaning only.
6. Before deciding, explicitly check these high-risk, easy-to-miss changes and treat any of them as category 3:
   - Negation flips: a "not / no / never" (or its source-language equivalent) present on one side but missing on the other (e.g. "which is NOT a demand" -> "which is a demand").
   - Quantity / condition changes: numbers, percentages, quantifiers ("all / some / at least / only / most"), or time / conditional phrases ("before / after / if / except") that differ between source and translation.
   - Relation changes: same referents but changed role structure (e.g. "sold to" vs "sold by", "caused by" vs "caused").
   A fluent, well-formed English sentence can still be wrong this way — do NOT mark OK just because the translation reads naturally.


Now evaluate the following item:

question: {question}
translation: {translation}
"""


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_export(path: Path):
    rows = []
    for row in read_jsonl(path):
        rows.append(row)
    return rows


def canonical_model_name(model: str) -> str:
    if model in {"gpt-30-samples", "gpt4o", "gpt"}:
        return "gpt"
    return model


def select_gpt_rows(gpt_rows, samples_per_language=None, samples_per_dataset=None, sample_seed=0):
    if samples_per_language is not None and samples_per_dataset is not None:
        raise ValueError("Specify only one of samples_per_language or samples_per_dataset.")

    rng = random.Random(sample_seed)

    if samples_per_dataset is not None:
        by_dataset = defaultdict(list)
        for row in gpt_rows:
            by_dataset[row["dataset"]].append(row)
        selected = []
        for dataset in DATASETS:
            rows = by_dataset.get(dataset, [])
            if len(rows) <= samples_per_dataset:
                selected.extend(rows)
            else:
                selected.extend(rng.sample(rows, samples_per_dataset))
        return selected

    if samples_per_language is None:
        return list(gpt_rows)

    by_key = defaultdict(list)
    for row in gpt_rows:
        by_key[(row["dataset"], row["language"])].append(row)
    selected = []
    for key in sorted(by_key):
        rows = by_key[key]
        if len(rows) <= samples_per_language:
            selected.extend(rows)
        else:
            selected.extend(rng.sample(rows, samples_per_language))
    return selected


def build_subset_index(gpt_rows):
    wanted = defaultdict(set)
    for row in gpt_rows:
        key = (row["dataset"], row["language"])
        wanted[key].add(str(row["example_id"]))
    return wanted


def filter_rows(rows, wanted):
    kept = []
    for row in rows:
        key = (row["dataset"], row["language"])
        if str(row["example_id"]) in wanted.get(key, set()):
            kept.append(row)
    return kept


def prompt_for(row):
    return PROMPT_TEMPLATE.format(question=row["x_l"], translation=row["x_en"])


def _response_error_details(resp: requests.Response, max_len: int = 500) -> str:
    status = getattr(resp, "status_code", None)
    detail = ""
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                detail = (
                    err.get("message")
                    or err.get("type")
                    or err.get("code")
                    or json.dumps(err, ensure_ascii=False)
                )
            else:
                detail = json.dumps(payload, ensure_ascii=False)
        else:
            detail = str(payload)
    except Exception:
        detail = (getattr(resp, "text", "") or "").strip()
    detail = " ".join(detail.split())
    if len(detail) > max_len:
        detail = detail[:max_len] + "..."
    if detail:
        return f"HTTP {status}: {detail}"
    return f"HTTP {status}"


class OpenAIResponsesJudge:
    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/responses",
        timeout_s: int = 300,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ):
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

    def _headers(self):
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _extract_text(self, data: dict) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            parts = []
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts).strip()
        raise RuntimeError("Responses API returned no output_text.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        payload = {
            "model": self.model_name,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_new_tokens,
        }
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                if resp.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(
                        _response_error_details(resp),
                        response=resp,
                    )
                if not resp.ok:
                    raise requests.exceptions.HTTPError(
                        _response_error_details(resp),
                        response=resp,
                    )
                return self._extract_text(resp.json())
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as err:
                last_err = err
                retryable = True
                if isinstance(err, requests.exceptions.HTTPError):
                    status = getattr(err.response, "status_code", None)
                    retryable = status in {408, 409, 429, 500, 502, 503, 504}
                if (not retryable) or (attempt >= self.max_retries):
                    raise
                sleep_s = self.retry_backoff_s * (2 ** attempt)
                LOGGER.warning(
                    "OpenAI responses request failed (%s). Retrying %d/%d in %.1fs",
                    err,
                    attempt + 1,
                    self.max_retries,
                    sleep_s,
                )
        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenAI responses request failed with unknown error")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        return [
            self.generate(
                prompt=p,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for p in prompts
        ]


def parse_judge_json(text):
    text = (text or "").strip()
    if not text:
        return {"error type": "PARSE_ERROR"}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"error type": "PARSE_ERROR"}


def normalize_error_type(value):
    if value is None:
        return "PARSE_ERROR"
    s = str(value).strip()
    if s.upper() == "OK":
        return "OK"
    if s in {"1", "2", "3", "4", "5"}:
        return s
    return "PARSE_ERROR"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_rows(rows):
    by_dataset = {}
    dataset_groups = defaultdict(list)
    for row in rows:
        dataset_groups[row["dataset"]].append(row)
    for dataset, ds_rows in sorted(dataset_groups.items()):
        ctr = Counter(r["error_type"] for r in ds_rows)
        by_dataset[dataset] = {
            "count": len(ds_rows),
            "error_type_counts": dict(sorted(ctr.items())),
        }
    ctr = Counter(r["error_type"] for r in rows)
    return {
        "count": len(rows),
        "error_type_counts": dict(sorted(ctr.items())),
        "by_dataset": by_dataset,
    }


def row_key(row) -> Tuple[str, str, str]:
    return (row["dataset"], row["language"], str(row["example_id"]))


def load_existing_judgments(path: Path) -> Dict[Tuple[str, str, str], dict]:
    existing = {}
    for row in read_jsonl(path) or []:
        existing[row_key(row)] = row
    return existing


def write_dataset_summary(dataset_dir: Path) -> List[dict]:
    rows = list(read_jsonl(dataset_dir / "judgments.jsonl") or [])
    write_json(dataset_dir / "summary.json", summarize_rows(rows))
    return rows


def write_model_summary(out_root: Path, model: str) -> None:
    model_rows = []
    for dataset in DATASETS:
        dataset_dir = out_root / model / dataset
        model_rows.extend(list(read_jsonl(dataset_dir / "judgments.jsonl") or []))
    write_json(out_root / model / "summary.json", summarize_rows(model_rows))


def run_model(model, rows, out_root, engine, max_new_tokens, temperature, top_p):
    for dataset in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        dataset_dir = out_root / model / dataset
        judgments_path = dataset_dir / "judgments.jsonl"
        existing = load_existing_judgments(judgments_path)
        pending_rows = [r for r in ds_rows if row_key(r) not in existing]
        LOGGER.info(
            "audit start model=%s dataset=%s total=%d existing=%d pending=%d",
            model,
            dataset,
            len(ds_rows),
            len(existing),
            len(pending_rows),
        )
        if not pending_rows:
            write_dataset_summary(dataset_dir)
            continue
        new_rows = []
        prompts = [prompt_for(r) for r in pending_rows]
        raw_outputs = engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for row, prompt, raw in zip(pending_rows, prompts, raw_outputs):
            parsed = parse_judge_json(raw)
            error_type = normalize_error_type(parsed.get("error type"))
            out_row = {
                "model": canonical_model_name(model),
                "dataset": dataset,
                "language": row["language"],
                "example_id": str(row["example_id"]),
                "error_type": error_type,
                "question": row["x_l"],
                "translation": row["x_en"],
                "selected_from_mode": row.get("selected_from_mode"),
                "judge_prompt": prompt,
                "judge_raw": raw,
                "judge_json": parsed,
            }
            new_rows.append(out_row)
        append_jsonl(judgments_path, new_rows)
        all_rows = write_dataset_summary(dataset_dir)
        LOGGER.info(
            "audit done model=%s dataset=%s wrote=%d total_saved=%d",
            model,
            dataset,
            len(new_rows),
            len(all_rows),
        )

    write_model_summary(out_root, model)


def validate_existing_manifest(out_dir: Path, manifest: dict) -> None:
    existing = read_json(out_dir / "manifest.json")
    if not existing:
        return

    comparable_keys = (
        "judge_model",
        "datasets",
        "models",
        "selection_rule",
        "selection",
        "generation",
    )
    existing_subset = {k: existing.get(k) for k in comparable_keys}
    requested_subset = {k: manifest.get(k) for k in comparable_keys}
    if existing_subset != requested_subset:
        raise RuntimeError(
            f"Existing manifest at {out_dir / 'manifest.json'} does not match requested selection/config. "
            "Use a different --out-dir or remove the old output directory."
        )


def main():
    ap = argparse.ArgumentParser(description="Audit MT1 translations with an OpenAI judge model.")
    ap.add_argument("--repo-root", default="/gscratch/stf/arnav/mt-llm-mt/Multiling-reasoning")
    ap.add_argument(
        "--exports-dir",
        default="results/analysis/mt1_translations",
        help="Directory containing *_oe_mt1_translations.jsonl exports.",
    )
    ap.add_argument(
        "--out-dir",
        default="results/analysis/mt1_translation_audit_gpt54mini",
        help="Output directory for judgments and summaries.",
    )
    ap.add_argument("--judge-model", default="gpt-5.4-mini")
    ap.add_argument(
        "--api-mode",
        choices=("responses", "chat_completions"),
        default="responses",
        help="OpenAI API surface to use for the judge model. Default: responses.",
    )
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for random subset selection. Default: 0.",
    )
    ap.add_argument(
        "--samples-per-language",
        type=int,
        default=None,
        help="Take the first N GPT rows per dataset/language, then align llama and mistral to the same rows. Default: 30 unless --samples-per-dataset is used.",
    )
    ap.add_argument(
        "--samples-per-dataset",
        type=int,
        default=None,
        help="Alternative selection mode: take the first N GPT rows per dataset total, then align llama and mistral to those rows.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    repo_root = Path(args.repo_root).resolve()
    exports_dir = Path(args.exports_dir)
    if not exports_dir.is_absolute():
        exports_dir = repo_root / exports_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    gpt_all_rows = load_export(exports_dir / f"{EXPORT_MODEL_FILES['gpt']}_oe_mt1_translations.jsonl")
    samples_per_language = args.samples_per_language
    if samples_per_language is None and args.samples_per_dataset is None:
        samples_per_language = 30
    gpt_rows = select_gpt_rows(
        gpt_all_rows,
        samples_per_language=samples_per_language,
        samples_per_dataset=args.samples_per_dataset,
        sample_seed=args.sample_seed,
    )
    wanted = build_subset_index(gpt_rows)

    model_to_rows = {"gpt": gpt_rows}
    for model in ("llama", "mistral"):
        rows = load_export(exports_dir / f"{EXPORT_MODEL_FILES[model]}_oe_mt1_translations.jsonl")
        model_to_rows[model] = filter_rows(rows, wanted)

    selection = {}
    if args.samples_per_dataset is not None:
        selection = {
            "mode": "per_dataset",
            "samples_per_dataset": args.samples_per_dataset,
            "sample_seed": args.sample_seed,
        }
        selection_rule = "Use a seeded random sample of N rows per dataset from gpt; filter llama and mistral to the same dataset/language/example_id rows."
    else:
        selection = {
            "mode": "per_language",
            "samples_per_language": samples_per_language,
            "sample_seed": args.sample_seed,
        }
        selection_rule = "Use a seeded random sample of N rows per dataset/language from gpt; filter llama and mistral to the same dataset/language/example_id rows."

    manifest = {
        "judge_model": args.judge_model,
        "api_mode": args.api_mode,
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "selection_rule": selection_rule,
        "selection": selection,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "available_gpt_rows": len(gpt_all_rows),
        "row_counts": {model: len(rows) for model, rows in model_to_rows.items()},
    }
    validate_existing_manifest(out_dir, manifest)
    write_json(out_dir / "manifest.json", manifest)

    if args.api_mode == "responses":
        engine = OpenAIResponsesJudge(
            model_name=args.judge_model,
            api_key_env=args.api_key_env,
            timeout_s=args.timeout_s,
        )
    else:
        engine = OpenAIEngine(
            model_name=args.judge_model,
            api_key_env=args.api_key_env,
            timeout_s=args.timeout_s,
        )
        engine.load_model()
    try:
        for model in MODELS:
            run_model(
                model,
                model_to_rows[model],
                out_dir,
                engine,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
    finally:
        try:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
