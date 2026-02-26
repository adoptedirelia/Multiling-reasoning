import argparse
import json
import os
from typing import Dict


def load_metrics(root_dir: str) -> Dict:
    results: Dict[str, Dict[str, Dict]] = {}
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "metrics.json" not in filenames:
            continue
        metrics_path = os.path.join(dirpath, "metrics.json")
        rel = os.path.relpath(dirpath, root_dir)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            # expect <tag>/<lang>
            continue
        lang = parts[-1]
        tag = "/".join(parts[:-1])
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        results.setdefault(tag, {})[lang] = metrics
    return results


def main():
    ap = argparse.ArgumentParser(description="Aggregate error_sim metrics into one JSON.")
    ap.add_argument(
        "--metrics_dir",
        default="results/error_sim/metrics",
        help="Root metrics directory (default: results/error_sim/metrics)",
    )
    ap.add_argument(
        "--output",
        default="results/error_sim/metrics/aggregate.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    data = {
        "metrics_root": args.metrics_dir,
        "results": load_metrics(args.metrics_dir),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
