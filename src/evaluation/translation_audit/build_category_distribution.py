#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict


CATEGORIES = ("OK", "1", "2", "3", "4", "5")
MODELS = ("gpt", "llama", "mistral")


def load_counts(model_dir: Path) -> Dict:
    counts = Counter()
    total = 0
    for path in sorted(model_dir.rglob("judgments.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts[str(row["error_type"])] += 1
                total += 1
    return {
        "count": total,
        "counts": {cat: counts.get(cat, 0) for cat in CATEGORIES},
        "percentages": {
            cat: round((counts.get(cat, 0) / total) * 100, 2) if total else 0.0
            for cat in CATEGORIES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translation-audit category distribution JSON.")
    parser.add_argument("--root", required=True, help="Judgment root containing gpt/llama/mistral subdirs.")
    parser.add_argument("--out", default="", help="Output JSON path. Defaults to <root>/category_distribution.json.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve() if args.out else root / "category_distribution.json"

    payload = {"categories": list(CATEGORIES), "models": {}}
    for model in MODELS:
        payload["models"][model] = load_counts(root / model)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
