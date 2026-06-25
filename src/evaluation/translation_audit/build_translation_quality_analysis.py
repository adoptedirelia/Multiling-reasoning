#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "results" / "translation_quality_analysis"
FULL_ROOT = ANALYSIS_ROOT / "full_llm_judgements"
OUT_DIR = ANALYSIS_ROOT / "tables"
SUPPLEMENTAL_COUNTS_PATH = OUT_DIR / "supplemental_counts.json"

MODELS = [
    ("gpt", "GPT-4o-mini"),
    ("llama", "Llama-3.1-8B-Instruct"),
    ("mistral", "Mistral-7B-Instruct-v0.3"),
]

CATEGORIES = ["OK", "1", "2", "3", "4", "5"]
DISPLAY_ORDER = [
    ("OK", r"\textsc{OK}"),
    ("1", r"\textsc{Struct.}"),
    ("2", r"\textsc{Entity}"),
    ("3", r"\textsc{Event}"),
    ("4", r"\textsc{Cultural}"),
    ("5", r"\textsc{Halluc.}"),
]


def normalize_error_type(value) -> str:
    if value is None:
        return "PARSE_ERROR"
    text = str(value).strip()
    if text.upper() == "OK":
        return "OK"
    if text in {"1", "2", "3", "4", "5"}:
        return text
    return text


def load_judgment_counts(model_root: Path) -> Tuple[int, Counter]:
    counts = Counter()
    total = 0
    for path in sorted(model_root.rglob("judgments.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                error_type = normalize_error_type(row.get("error_type"))
                counts[error_type] += 1
                total += 1
    return total, counts


def load_supplemental_counts() -> Dict:
    if not SUPPLEMENTAL_COUNTS_PATH.exists():
        return {}
    return json.loads(SUPPLEMENTAL_COUNTS_PATH.read_text(encoding="utf-8"))


def merge_counts(base_total: int, base_counts: Counter, supplemental: Dict) -> Tuple[int, Counter]:
    merged_counts = Counter(base_counts)
    extra_total = int(supplemental.get("count", 0))
    for category in CATEGORIES:
        merged_counts[category] += int(supplemental.get("counts", {}).get(category, 0))
    return base_total + extra_total, merged_counts


def build_payload() -> Dict:
    supplemental_by_model = load_supplemental_counts().get("models", {})
    payload = {
        "categories": CATEGORIES,
        "models": {},
    }
    for model_key, model_label in MODELS:
        base_total, base_counts = load_judgment_counts(FULL_ROOT / model_key)
        merged_total, merged_counts = merge_counts(
            base_total,
            base_counts,
            supplemental_by_model.get(model_key, {}),
        )
        payload["models"][model_key] = {
            "model_label": model_label,
            "count": merged_total,
            "counts": {category: merged_counts.get(category, 0) for category in CATEGORIES},
            "percentages": {
                category: round((merged_counts.get(category, 0) / merged_total) * 100, 2)
                if merged_total
                else 0.0
                for category in CATEGORIES
            },
        }
    return payload


def write_json(payload: Dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "translation_error_distribution.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def write_tex(payload: Dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "translation_error_distribution.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & \\textsc{OK} & \\textsc{Struct.} & \\textsc{Entity} & \\textsc{Event} & \\textsc{Cultural} & \\textsc{Halluc.} \\\\",
        "\\midrule",
    ]
    for model_key, model_label in MODELS:
        percentages = payload["models"][model_key]["percentages"]
        values = [fmt_pct(percentages[category]) for category, _ in DISPLAY_ORDER]
        lines.append(
            f"{model_label} & {values[0]} & {values[1]} & {values[2]} & {values[3]} & {values[4]} & {values[5]} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Distribution over all datasets of GPT-5.4-mini translation-judged error labels (\\%). \\textsc{Struct.} denotes structural/formatting error, \\textsc{Entity} denotes referent/entity substitution, \\textsc{Event} denotes event/constraint distortion, \\textsc{Cultural} denotes cultural/local-term mistranslation, and \\textsc{Halluc.} denotes hallucination/over-answering. The most prominent errors across all models are structural/formatting errors, event/constraint distortions, and hallucination/over-answering.}",
            "\\label{tab:mt1_audit_merged}",
            "\\end{table}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    payload = build_payload()
    json_path = write_json(payload)
    tex_path = write_tex(payload)
    print(json_path)
    print(tex_path)


if __name__ == "__main__":
    main()
