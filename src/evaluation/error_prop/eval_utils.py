import json
import os
import subprocess
from typing import Dict


def write_predictions_jsonl(path: str, preds: Dict[str, str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex_id, pred in preds.items():
            rec = {
                "example_id": ex_id,
                "prediction": pred,
                "binary_answer": None,
                "no_answer_prob": 0.0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_mkqa_eval(mkqa_dir: str, annotation_file: str, predictions_file: str, language: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "python",
        os.path.join(mkqa_dir, "mkqa_eval.py"),
        "--annotation_file",
        annotation_file,
        "--predictions_file",
        predictions_file,
        "--language",
        language,
        "--out-dir",
        out_dir,
    ]
    env = os.environ.copy()
    env["MKQA_NO_PLOTS"] = "1"
    subprocess.check_call(cmd, env=env)
