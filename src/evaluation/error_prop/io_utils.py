import csv
import json
import os
from datetime import datetime
from typing import Dict, List


def setup_logging(logs_dir: str, run_name: str) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"{run_name}_{timestamp}.log")
    return log_path


def append_row_csv(path: str, fieldnames: List[str], row: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_constraints(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        constraints = raw
    elif isinstance(raw, dict):
        constraints = raw.get("constraints", [])
    else:
        raise ValueError(f"Unsupported constraints format in {path}")

    for c in constraints:
        if "id" not in c or "text" not in c:
            raise ValueError(f"Each constraint must contain id and text. Bad entry: {c}")
    return constraints
