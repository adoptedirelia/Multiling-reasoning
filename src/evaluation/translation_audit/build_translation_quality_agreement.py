#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path("/gscratch/stf/arnav/mt-llm-mt/Multiling-reasoning")
ANALYSIS_ROOT = REPO_ROOT / "results" / "translation_quality_analysis"
WITH_HUMAN_ROOT = ANALYSIS_ROOT / "with_human_judge"
FULL_105_PATH = WITH_HUMAN_ROOT / "full_105_human_llm_annotations.json"
OUT_DIR = ANALYSIS_ROOT / "tables"

LABELS = ["OK", "1", "2", "3", "4", "5"]
RATERS = {
    "human1": "human1",
    "human2": "human2",
    "llm": "GPT-5.4-mini",
}


def normalize_label(value: str) -> str:
    text = str(value).strip().replace("ΟΚ", "OK")
    return "OK" if text.upper() == "OK" else text


def load_rows() -> List[Dict]:
    rows = json.loads(FULL_105_PATH.read_text(encoding="utf-8"))
    for row in rows:
        row["human1"] = normalize_label(row["human1"])
        row["human2"] = normalize_label(row["human2"])
        row["llm"] = normalize_label(row["llm"])
    return rows


def cohens_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in LABELS)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa_three_way(rows: List[Dict]) -> float:
    n_items = len(rows)
    n_raters = 3
    label_probs = {label: 0.0 for label in LABELS}
    per_item_agreement = []

    for row in rows:
        labels = [row["human1"], row["human2"], row["llm"]]
        counts = Counter(labels)
        for label in LABELS:
            label_probs[label] += counts[label]
        per_item_agreement.append(
            (sum(count * count for count in counts.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )

    for label in LABELS:
        label_probs[label] /= (n_items * n_raters)

    p_bar = sum(per_item_agreement) / n_items
    p_e = sum(prob * prob for prob in label_probs.values())
    if p_e == 1.0:
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def round3(value: float) -> float:
    return round(value + 1e-12, 3)


def build_payload(rows: List[Dict]) -> Dict:
    human1 = [row["human1"] for row in rows]
    human2 = [row["human2"] for row in rows]
    llm = [row["llm"] for row in rows]

    pairwise = {
        "human1_vs_human2": {
            "rater_a": RATERS["human1"],
            "rater_b": RATERS["human2"],
            "cohen_kappa": round3(cohens_kappa(human1, human2)),
        },
        "human1_vs_llm": {
            "rater_a": RATERS["human1"],
            "rater_b": RATERS["llm"],
            "cohen_kappa": round3(cohens_kappa(human1, llm)),
        },
        "human2_vs_llm": {
            "rater_a": RATERS["human2"],
            "rater_b": RATERS["llm"],
            "cohen_kappa": round3(cohens_kappa(human2, llm)),
        },
    }

    return {
        "annotation_source": str(FULL_105_PATH),
        "n_items": len(rows),
        "label_space": LABELS,
        "pairwise": pairwise,
        "three_way": {
            "raters": [RATERS["human1"], RATERS["human2"], RATERS["llm"]],
            "fleiss_kappa": round3(fleiss_kappa_three_way(rows)),
        },
    }


def write_json(payload: Dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "translation_agreement_kappas.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def write_tex(payload: Dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "translation_agreement_kappas.tex"
    p = payload["pairwise"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Agreement pair & $\\kappa$ \\\\",
        "\\midrule",
        f"{RATERS['human1']} vs. {RATERS['human2']} & {p['human1_vs_human2']['cohen_kappa']:.3f} \\\\",
        f"{RATERS['human1']} vs. {RATERS['llm']} & {p['human1_vs_llm']['cohen_kappa']:.3f} \\\\",
        f"{RATERS['human2']} vs. {RATERS['llm']} & {p['human2_vs_llm']['cohen_kappa']:.3f} \\\\",
        f"Three-way Fleiss' $\\kappa$ & {payload['three_way']['fleiss_kappa']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Translation-audit agreement on the full annotated set ($n={payload['n_items']}$ items). Pairwise values are Cohen's $\\kappa$; the three-rater value is Fleiss' $\\kappa$.}}",
        "\\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    rows = load_rows()
    payload = build_payload(rows)
    json_path = write_json(payload)
    tex_path = write_tex(payload)
    print(json_path)
    print(tex_path)


if __name__ == "__main__":
    main()
