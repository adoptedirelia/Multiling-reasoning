#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .final_ctx_std_tests import (
    compute_ctx_std_tests_for_model,
    write_results_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run binomial and Wilcoxon tests for context vs standard final metrics")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "results" / "significance" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compute_ctx_std_tests_for_model(repo_root, args.model)
    write_results_json(out_dir / "ctx_vs_std_sigtests.json", rows)
    print(out_dir / "ctx_vs_std_sigtests.json")


if __name__ == "__main__":
    main()
