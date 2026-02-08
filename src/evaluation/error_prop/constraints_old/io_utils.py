import json
import os
from datetime import datetime
from typing import Dict, List


def setup_logging(logs_dir: str, run_name: str) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"{run_name}_{timestamp}.log")
    return log_path


def save_results_json(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


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
