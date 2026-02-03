#!/usr/bin/env python3
import json
import pathlib
from typing import Dict, List, Tuple


BASE = pathlib.Path("/gscratch/ark/arnav/mt-llm-mt/Multiling-reasoning/data_transfer")
FILES = {
    "answer": BASE / "culture_scores.json",
    "ground_truth": BASE / "culture_scores_gt.json",
    "answer_with_gt": BASE / "culture_scores_mt.json",
}


def _load(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def _collect_keys(rows: List[Dict]) -> Tuple[List[str], List[str]]:
    culture_keys = []
    correct_keys = []
    for k in rows[0].keys():
        if k.endswith("_culture"):
            culture_keys.append(k)
        elif k.endswith("_correctness"):
            correct_keys.append(k)
    return culture_keys, correct_keys


def _avg(rows: List[Dict], key: str) -> float:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    for label, path in FILES.items():
        rows = _load(path)
        culture_keys, correct_keys = _collect_keys(rows)
        langs = sorted({str(r.get("language", "")).strip() for r in rows})
        print(f"== {label} ==")
        for lang in langs:
            subset = [r for r in rows if str(r.get("language", "")).strip() == lang]
            if not subset:
                continue
            print(f"-- {lang} --")
            for k in culture_keys:
                print(f"{k}: {_avg(subset, k):.4f}")
            for k in correct_keys:
                print(f"{k}: {_avg(subset, k):.4f}")
        print()


if __name__ == "__main__":
    main()
