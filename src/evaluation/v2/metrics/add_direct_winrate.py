import argparse
import json
import logging
import os
from typing import Dict, List

from .runner import run_metrics

LOGGER = logging.getLogger(__name__)


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_baseline_standard(row: Dict) -> bool:
    s = str(row.get("slice", "")).strip().lower()
    if s == "baseline/standard":
        return True
    eg = str(row.get("error_group", "")).strip().lower()
    cm = str(row.get("cascade_mode", row.get("cascade", ""))).strip().lower()
    return eg == "baseline" and cm == "standard"


def _build_direct_row(template_row: Dict, lang: str, ex_id: str, pred: str) -> Dict:
    return {
        "run_name": str(template_row.get("run_name", "")),
        "lang": lang,
        "example_id": ex_id,
        "error_group": "baseline",
        "error_type": None,
        "cascade": "direct",
        "cascade_mode": "direct",
        "slice": "baseline/direct",
        "x_l": template_row.get("x_l", ""),
        "x_en": None,
        "r_en": None,
        "y_en": None,
        "y_l": pred,
        "prediction": pred,
        "x_en_err": None,
        "r_en_err": None,
        "y_en_err": None,
    }


def add_direct_winrate(
    config_path: str,
    base_preds_dir: str,
    direct_preds_dir: str,
    existing_winrate_json: str,
    out_json: str,
    langs: List[str],
    prediction_field: str,
) -> str:
    LOGGER.info(
        "add_direct_winrate start config=%s base_preds=%s direct_preds=%s existing=%s out=%s langs=%s",
        config_path,
        base_preds_dir,
        direct_preds_dir,
        existing_winrate_json,
        out_json,
        langs,
    )
    out_stem = os.path.splitext(os.path.basename(out_json))[0] or "tmp_direct_only"
    temp_pred_dir = os.path.join("results", "v2", "winrate_inputs", f"{out_stem}_tmp_preds")
    os.makedirs(temp_pred_dir, exist_ok=True)

    for lang in langs:
        base_path = os.path.join(base_preds_dir, f"{lang}.jsonl")
        direct_path = os.path.join(direct_preds_dir, f"{lang}.jsonl")
        if not os.path.exists(base_path):
            LOGGER.info("lang=%s skipped: missing base path %s", lang, base_path)
            continue
        if not os.path.exists(direct_path):
            LOGGER.info("lang=%s skipped: missing direct path %s", lang, direct_path)
            continue

        base_rows = _load_jsonl(base_path)
        direct_rows = _load_jsonl(direct_path)

        std_rows = [r for r in base_rows if _is_baseline_standard(r)]
        if not std_rows:
            LOGGER.info("lang=%s skipped: no baseline/standard rows", lang)
            continue

        std_by_ex = {str(r.get("example_id", "")): r for r in std_rows if str(r.get("example_id", ""))}
        out_rows = list(std_rows)
        for dr in direct_rows:
            ex_id = str(dr.get("example_id", ""))
            pred = dr.get("prediction")
            if not ex_id or not isinstance(pred, str) or not pred.strip():
                continue
            template = std_by_ex.get(ex_id)
            if template is None:
                continue
            out_rows.append(_build_direct_row(template, lang, ex_id, pred))

        _write_jsonl(os.path.join(temp_pred_dir, f"{lang}.jsonl"), out_rows)
        LOGGER.info("lang=%s direct merge rows_written=%d", lang, len(out_rows))

    temp_metrics = os.path.join("results", "v2", "winrate_inputs", f"{out_stem}_tmp_metrics.json")
    run_metrics(
        config_path=config_path,
        predictions_jsonl=temp_pred_dir,
        out_json=temp_metrics,
        prediction_field=prediction_field,
        methods_csv="win_rate",
    )
    LOGGER.info("computed temporary direct-only metrics at %s", temp_metrics)

    base_metrics = _load_json(existing_winrate_json)
    direct_metrics = _load_json(temp_metrics)

    src_slice = direct_metrics.get("slices", {}).get("baseline/standard", {}).get("by_language", {})
    dst_slices = base_metrics.setdefault("slices", {})
    dst_baseline = dst_slices.setdefault("baseline/standard", {}).setdefault("by_language", {})

    for lang, row in src_slice.items():
        w = row.get("standard_vs_direct_win_rate")
        if w is None:
            continue
        dst_row = dst_baseline.setdefault(lang, {})
        dst_row["standard_vs_direct_win_rate"] = w

    _write_json(out_json, base_metrics)
    LOGGER.info("add_direct_winrate done: wrote %s", out_json)
    return out_json


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Add standard-vs-direct win rates to existing winrate JSON")
    ap.add_argument("--config", required=True)
    ap.add_argument("--base_preds_dir", required=True)
    ap.add_argument("--direct_preds_dir", required=True)
    ap.add_argument("--existing_winrate_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--langs", required=True, help="comma-separated language codes")
    ap.add_argument("--prediction_field", default="y_l")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out = add_direct_winrate(
        config_path=args.config,
        base_preds_dir=args.base_preds_dir,
        direct_preds_dir=args.direct_preds_dir,
        existing_winrate_json=args.existing_winrate_json,
        out_json=args.out_json,
        langs=langs,
        prediction_field=args.prediction_field,
    )
    print(f"Wrote updated winrate JSON with direct-eval win rates: {out}")


if __name__ == "__main__":
    main()
