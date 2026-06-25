#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_LEVEL_PY = REPO_ROOT / "src" / "eval" / "resource_level.py"
RESOURCE_FEATURES_CSV = REPO_ROOT / "std_cxt_e2e_language_features.csv"
HIGH_THRESHOLD = 16.0
MID_THRESHOLD = 12.0


DATASET_KIND = {
    "mmlu": "mcq",
    "global-mmlu": "mcq",
    "global_mmlu": "mcq",
    "belebele": "mcq",
    "global-piqa-mc": "mcq",
    "global_piqa_mc": "mcq",
    "piqa": "mcq",
    "mcsqa": "mcq",
    "mgsm": "math",
    "mmath": "math",
}

RESOURCE_DATASET_KEY = {
    "mmlu": "global_mmlu",
    "global-mmlu": "global_mmlu",
    "global_mmlu": "global_mmlu",
    "belebele": "belebele",
    "global-piqa-mc": "piqa",
    "global_piqa_mc": "piqa",
    "piqa": "piqa",
    "mcsqa": "mcsqa",
    "mgsm": "mgsm",
    "mmath": "mmath",
}

MCQ_LABELS = ("A", "B", "C", "D", "E")


def _tier(score: Optional[float]) -> str:
    if score is None or score < MID_THRESHOLD:
        return "low"
    if score < HIGH_THRESHOLD:
        return "mid"
    return "high"


def _load_resource_level_constants() -> Dict[str, Dict[str, str]]:
    text = RESOURCE_LEVEL_PY.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n", ""
    )
    module = ast.parse(text, filename=str(RESOURCE_LEVEL_PY))
    constants = {}
    needed = {
        "_GLOBAL_MMLU",
        "_BELEBELE",
        "_MCSQA",
        "_MMATH",
        "_PIQA",
        "_MGSM",
        "_AYA",
        "_MKQA",
        "_BLEND",
    }
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in needed:
                constants[target.id] = ast.literal_eval(node.value)
    missing = needed - set(constants)
    if missing:
        raise ValueError(f"Missing resource-level constants: {sorted(missing)}")
    return constants


def _load_resource_score_lookups():
    flores_to_score = {}
    code2_to_score = {}
    code3_to_score = {}
    with RESOURCE_FEATURES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flores = str(row["language"]).strip().lower()
            code2 = str(row["language_code"]).strip().lower()
            code3 = str(row["resource_lookup_code"]).strip().lower()
            raw = row["resource_score_log_pages"]
            try:
                score = float(raw)
            except (TypeError, ValueError):
                score = None

            flores_to_score[flores] = score
            if code2 not in code2_to_score or (
                code2_to_score[code2] is None and score is not None
            ):
                code2_to_score[code2] = score
            if code3 not in code3_to_score or (
                code3_to_score[code3] is None and score is not None
            ):
                code3_to_score[code3] = score

    flores_to_score["us"] = flores_to_score.get("eng_latn")
    flores_to_score["gb"] = flores_to_score.get("eng_latn")
    return flores_to_score, code2_to_score, code3_to_score


RESOURCE_LEVEL_CONSTANTS = _load_resource_level_constants()
FLORES_TO_SCORE, CODE2_TO_SCORE, CODE3_TO_SCORE = _load_resource_score_lookups()


def _flores_score(code: str) -> Optional[float]:
    key = str(code).lower()
    score = FLORES_TO_SCORE.get(key)
    if score is not None:
        return score
    prefix = key.split("_")[0]
    score = FLORES_TO_SCORE.get(prefix)
    if score is not None:
        return score
    return CODE3_TO_SCORE.get(prefix)


def _build_resource_levels() -> Dict[str, Dict[str, str]]:
    def by_code2(lang_map):
        return {name: _tier(CODE2_TO_SCORE.get(code.lower())) for code, name in lang_map.items()}

    def by_code3(lang_map):
        return {name: _tier(CODE3_TO_SCORE.get(code.lower())) for code, name in lang_map.items()}

    def by_flores(lang_map):
        return {name: _tier(_flores_score(code)) for code, name in lang_map.items()}

    return {
        "global_mmlu": by_code2(RESOURCE_LEVEL_CONSTANTS["_GLOBAL_MMLU"]),
        "belebele": by_flores(RESOURCE_LEVEL_CONSTANTS["_BELEBELE"]),
        "mcsqa": by_code2(RESOURCE_LEVEL_CONSTANTS["_MCSQA"]),
        "mmath": by_code2(RESOURCE_LEVEL_CONSTANTS["_MMATH"]),
        "piqa": by_flores(RESOURCE_LEVEL_CONSTANTS["_PIQA"]),
        "mgsm": by_code2(RESOURCE_LEVEL_CONSTANTS["_MGSM"]),
        "aya": by_code3(RESOURCE_LEVEL_CONSTANTS["_AYA"]),
        "mkqa": by_flores(RESOURCE_LEVEL_CONSTANTS["_MKQA"]),
        "blend": by_flores(RESOURCE_LEVEL_CONSTANTS["_BLEND"]),
    }


RESOURCE_LEVELS = _build_resource_levels()


def get_resource_level(dataset: str, language: str) -> str:
    key = dataset.lower().replace("-", "_").replace(" ", "_")
    return RESOURCE_LEVELS.get(key, {}).get(language, "low")


def _round(value: float, ndigits: int = 2) -> float:
    return round(float(value), ndigits)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_loose(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"[\s\.,;:!?\"'`“”‘’\(\)\[\]\{\}<>\-_/\\]+", "", text)
    return text


def _extract_prediction_text(row: Dict) -> str:
    if isinstance(row.get("result"), dict):
        return str((row["result"] or {}).get("answer", "") or "")
    if isinstance(row.get("mt2_result"), dict):
        return str((row["mt2_result"] or {}).get("answer", "") or "")
    if isinstance(row.get("llm_result"), dict):
        return str((row["llm_result"] or {}).get("answer", "") or "")
    return str(row.get("prediction", "") or "")


def _extract_choice_label(prediction: str) -> Optional[str]:
    raw = unicodedata.normalize("NFKC", str(prediction or "")).strip()
    if not raw:
        return None

    # Conservative explicit-choice patterns only. Avoid treating article "a" as option A.
    patterns = [
        r"^\s*[\(\[]?\s*([A-E])\s*[\)\]\.\:\-]?\s*$",
        r"^\s*(?:option|choice|answer)\s*[:\-]?\s*[\(\[]?\s*([A-E])\s*[\)\]\.\:]?\s*$",
        r"^\s*(?:option|choice|answer)\s+([A-E])(?:\b|[\)\]\.\:])",
    ]
    for pattern in patterns:
        match = re.match(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _infer_mcq_label(prediction: str, record: Dict) -> Optional[str]:
    explicit = _extract_choice_label(prediction)
    if explicit:
        return explicit

    pred_norm = _normalize_loose(prediction)
    if not pred_norm:
        return None

    option_map = {}
    for label in MCQ_LABELS:
        field = f"option_{label.lower()}"
        if field in record and record[field]:
            option_map[label] = _normalize_loose(record[field])

    exact_matches = [label for label, opt in option_map.items() if pred_norm == opt]
    if len(exact_matches) == 1:
        return exact_matches[0]

    substring_matches = [label for label, opt in option_map.items() if opt and opt in pred_norm]
    if len(substring_matches) == 1:
        return substring_matches[0]

    return None


def _normalize_math_answer(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", value)
    if boxed:
        value = boxed[-1]
    answer_match = re.search(r"<answer>(.*?)</answer>", value, flags=re.DOTALL | re.IGNORECASE)
    if answer_match:
        value = answer_match.group(1)
    value = value.strip().strip("$")
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value.lower()


def _is_correct_mcq(prediction: str, record: Dict) -> bool:
    gold_label = str(record.get("answer", "") or "").strip().upper()
    gold_text = str(
        record.get("answer_text")
        or record.get(f"option_{gold_label.lower()}", "")
        or ""
    )

    inferred = _infer_mcq_label(prediction, record)
    if inferred is not None:
        return inferred == gold_label

    pred_norm = _normalize_loose(prediction)
    gold_norm = _normalize_loose(gold_text)
    return bool(pred_norm and gold_norm and pred_norm == gold_norm)


def _is_correct_math(prediction: str, record: Dict) -> bool:
    pred_norm = _normalize_math_answer(prediction)
    gold_norm = _normalize_math_answer(str(record.get("answer", "") or ""))
    return bool(pred_norm and gold_norm and pred_norm == gold_norm)


def _dataset_kind(dataset_key: str) -> str:
    key = dataset_key.lower().replace("_", "-")
    if key not in DATASET_KIND:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")
    return DATASET_KIND[key]


def _resource_dataset_key(dataset_key: str) -> str:
    key = dataset_key.lower().replace("_", "-")
    return RESOURCE_DATASET_KEY[key]


def _flatten_dataset_blob(blob, split: str = "") -> List[Dict]:
    data = blob
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        chosen = split or ("test" if "test" in data else "dev" if "dev" in data else next(iter(data)))
        data = data[chosen]
    if isinstance(data, dict):
        flattened = []
        for language, rows in data.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                item = dict(row)
                item["language"] = language
                flattened.append(item)
        return flattened
    if isinstance(data, list):
        return [dict(row) for row in data]
    raise ValueError("Unsupported dataset JSON structure")


def _score_rows(dataset_key: str, dataset_records: List[Dict], rows: List[Dict]) -> Dict[str, List[int]]:
    kind = _dataset_kind(dataset_key)
    by_language: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        sample_idx = row.get("sample_idx", idx)
        if not isinstance(sample_idx, int):
            try:
                sample_idx = int(sample_idx)
            except Exception:
                continue
        if sample_idx < 0 or sample_idx >= len(dataset_records):
            continue

        record = dataset_records[sample_idx]
        language = str(row.get("language") or record.get("language") or "unknown")
        prediction = _extract_prediction_text(row)

        if kind == "mcq":
            correct = _is_correct_mcq(prediction, record)
        else:
            correct = _is_correct_math(prediction, record)

        by_language.setdefault(language, []).append(1 if correct else 0)
    return by_language


def _build_slice_payload(dataset_key: str, by_language_scores: Dict[str, List[int]]) -> Dict:
    resource_key = _resource_dataset_key(dataset_key)
    by_language = {}
    by_resource_acc: Dict[str, List[float]] = {"high": [], "mid": [], "low": []}
    by_resource_counts: Dict[str, int] = {"high": 0, "mid": 0, "low": 0}

    for language in sorted(by_language_scores):
        scores = by_language_scores[language]
        if not scores:
            continue
        accuracy = _round(100.0 * sum(scores) / len(scores))
        resource_level = get_resource_level(resource_key, language)
        by_language[language] = {
            "resource_level": resource_level,
            "count": len(scores),
            "accuracy": accuracy,
        }
        by_resource_acc.setdefault(resource_level, []).append(accuracy)
        by_resource_counts[resource_level] = by_resource_counts.get(resource_level, 0) + len(scores)

    overall_acc_values = [row["accuracy"] for row in by_language.values()]
    payload = {
        "overall": {
            "languages": len(by_language),
            "count": sum(row["count"] for row in by_language.values()),
            "accuracy": _round(sum(overall_acc_values) / len(overall_acc_values)) if overall_acc_values else 0.0,
        },
        "by_resource": {},
        "by_language": by_language,
    }

    for resource_level in ("high", "mid", "low"):
        accs = by_resource_acc.get(resource_level, [])
        payload["by_resource"][resource_level] = {
            "languages": len(accs),
            "count": by_resource_counts.get(resource_level, 0),
            "accuracy": _round(sum(accs) / len(accs)) if accs else 0.0,
        }
    return payload


def _load_rows(path: Path) -> List[Dict]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of result rows in {path}")
    return data


def export_metrics(
    *,
    dataset_key: str,
    model: str,
    dataset_json: Path,
    standard_json: Path,
    context_json: Path,
    out_json: Path,
    direct_json: Optional[Path] = None,
    dataset_split: str = "",
    run_name: str = "",
) -> Path:
    dataset_records = _flatten_dataset_blob(_load_json(dataset_json), split=dataset_split)
    slices = {
        "baseline/standard": _build_slice_payload(dataset_key, _score_rows(dataset_key, dataset_records, _load_rows(standard_json))),
        "baseline/context": _build_slice_payload(dataset_key, _score_rows(dataset_key, dataset_records, _load_rows(context_json))),
    }
    if direct_json:
        slices["baseline/direct"] = _build_slice_payload(
            dataset_key,
            _score_rows(dataset_key, dataset_records, _load_rows(direct_json)),
        )

    payload = {
        "run_name": run_name or f"{model}_{dataset_key}_accuracy_export",
        "dataset": dataset_key,
        "methods": ["accuracy"],
        "slices": slices,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_json


def export_single_slice_metrics(
    *,
    dataset_key: str,
    model: str,
    dataset_json: Path,
    input_json: Path,
    slice_name: str,
    out_json: Path,
    dataset_split: str = "",
    run_name: str = "",
) -> Path:
    dataset_records = _flatten_dataset_blob(_load_json(dataset_json), split=dataset_split)
    payload = {
        "run_name": run_name or f"{model}_{dataset_key}_{slice_name.replace('/', '_')}_accuracy_export",
        "dataset": dataset_key,
        "methods": ["accuracy"],
        "slices": {
            slice_name: _build_slice_payload(
                dataset_key,
                _score_rows(dataset_key, dataset_records, _load_rows(input_json)),
            )
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_json


def _run_manifest(manifest_path: Path) -> List[Path]:
    manifest = _load_json(manifest_path)
    jobs = manifest["jobs"] if isinstance(manifest, dict) and "jobs" in manifest else manifest
    if not isinstance(jobs, list):
        raise ValueError("Manifest must be a list or a dict with a 'jobs' list")

    outputs = []
    for job in jobs:
        if job.get("input_json") and job.get("slice_name"):
            outputs.append(
                export_single_slice_metrics(
                    dataset_key=job["dataset_key"],
                    model=job["model"],
                    dataset_json=Path(job["dataset_json"]),
                    input_json=Path(job["input_json"]),
                    slice_name=job["slice_name"],
                    out_json=Path(job["out_json"]),
                    dataset_split=job.get("dataset_split", ""),
                    run_name=job.get("run_name", ""),
                )
            )
        else:
            outputs.append(
                export_metrics(
                    dataset_key=job["dataset_key"],
                    model=job["model"],
                    dataset_json=Path(job["dataset_json"]),
                    standard_json=Path(job["standard_json"]),
                    context_json=Path(job["context_json"]),
                    direct_json=Path(job["direct_json"]) if job.get("direct_json") else None,
                    out_json=Path(job["out_json"]),
                    dataset_split=job.get("dataset_split", ""),
                    run_name=job.get("run_name", ""),
                )
            )
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Export pulled MCQ/math raw result JSONs into metrics.json files compatible with significance and resource violin plots."
    )
    parser.add_argument("--manifest", default="", help="Optional JSON manifest of export jobs.")
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset-json", default="")
    parser.add_argument("--dataset-split", default="", help="Optional split key when dataset JSON has nested train/dev/test maps.")
    parser.add_argument("--standard-json", default="")
    parser.add_argument("--context-json", default="")
    parser.add_argument("--direct-json", default="")
    parser.add_argument("--input-json", default="")
    parser.add_argument("--slice-name", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    outputs: List[Path]
    if args.manifest:
        outputs = _run_manifest(Path(args.manifest))
    else:
        if args.input_json or args.slice_name:
            required = {
                "--dataset-key": args.dataset_key,
                "--model": args.model,
                "--dataset-json": args.dataset_json,
                "--input-json": args.input_json,
                "--slice-name": args.slice_name,
                "--out": args.out,
            }
            missing = [flag for flag, value in required.items() if not value]
            if missing:
                raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

            outputs = [
                export_single_slice_metrics(
                    dataset_key=args.dataset_key,
                    model=args.model,
                    dataset_json=Path(args.dataset_json),
                    input_json=Path(args.input_json),
                    slice_name=args.slice_name,
                    out_json=Path(args.out),
                    dataset_split=args.dataset_split,
                    run_name=args.run_name,
                )
            ]
        else:
            required = {
                "--dataset-key": args.dataset_key,
                "--model": args.model,
                "--dataset-json": args.dataset_json,
                "--standard-json": args.standard_json,
                "--context-json": args.context_json,
                "--out": args.out,
            }
            missing = [flag for flag, value in required.items() if not value]
            if missing:
                raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

            outputs = [
                export_metrics(
                    dataset_key=args.dataset_key,
                    model=args.model,
                    dataset_json=Path(args.dataset_json),
                    standard_json=Path(args.standard_json),
                    context_json=Path(args.context_json),
                    direct_json=Path(args.direct_json) if args.direct_json else None,
                    out_json=Path(args.out),
                    dataset_split=args.dataset_split,
                    run_name=args.run_name,
                )
            ]

    for out_path in outputs:
        print(out_path)


if __name__ == "__main__":
    main()
