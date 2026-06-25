import argparse
import json
import os
from typing import Any, Dict, List, Optional

from .config import V2Config, load_config
from .loaders.registry import load_records_for_language


LANG_ALIASES = {
    "zh": "zh_cn",
    "cn": "zh_cn",
}


def _read_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is not a JSON list")
            return [r for r in data if isinstance(r, dict)]
        rows: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows


def _write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _exists_lang_file(pred_dir: str, lang: str) -> bool:
    return os.path.exists(os.path.join(pred_dir, f"{lang}.json")) or os.path.exists(
        os.path.join(pred_dir, f"{lang}.jsonl")
    )


def _resolve_input(pred_dir: str, lang: str) -> str:
    json_path = os.path.join(pred_dir, f"{lang}.json")
    if os.path.exists(json_path):
        return json_path
    jsonl_path = os.path.join(pred_dir, f"{lang}.jsonl")
    if os.path.exists(jsonl_path):
        return jsonl_path
    raise FileNotFoundError(f"Neither {json_path} nor {jsonl_path} exists")


def _resolve_lang(lang_arg: str, pred_dir: str, cfg: V2Config) -> str:
    if _exists_lang_file(pred_dir, lang_arg) and lang_arg in cfg.dataset.langs:
        return lang_arg

    alias = LANG_ALIASES.get(lang_arg)
    if alias and _exists_lang_file(pred_dir, alias) and alias in cfg.dataset.langs:
        return alias

    if _exists_lang_file(pred_dir, lang_arg):
        return lang_arg

    if alias and _exists_lang_file(pred_dir, alias):
        return alias

    return lang_arg


def _default_prefix(cfg: V2Config) -> str:
    marker = "v2_refresh_gpt_"
    if cfg.run_name.startswith(marker):
        return cfg.run_name[len(marker) :]
    return cfg.dataset.dataset_type


def export_baseline(
    lang: str,
    pred_dir: str,
    config_path: str,
    out_prefix: Optional[str] = None,
) -> str:
    cfg = load_config(config_path)
    effective_lang = _resolve_lang(lang, pred_dir, cfg)
    out_rows = _export_one_lang(effective_lang, pred_dir, cfg)

    prefix = out_prefix or _default_prefix(cfg)
    out_path = os.path.join(pred_dir, f"{prefix}_{lang}.json")
    _write_json(out_path, out_rows)
    return out_path


def _export_one_lang(effective_lang: str, pred_dir: str, cfg: V2Config) -> List[Dict[str, Any]]:
    in_path = _resolve_input(pred_dir, effective_lang)
    rows = _read_records(in_path)
    baseline = [r for r in rows if r.get("error_group") == "baseline"]

    gold_records = load_records_for_language(cfg.dataset, effective_lang)
    gold_by_id = {str(r["example_id"]): list(r.get("y_l_gold", [])) for r in gold_records}

    out_rows: List[Dict[str, Any]] = []
    missing = []
    for r in baseline:
        ex_id = str(r.get("example_id", ""))
        if ex_id not in gold_by_id:
            missing.append(ex_id)
            continue
        mode = str(r.get("cascade_mode", ""))
        is_direct = mode == "direct"
        out_rows.append(
            {
                "language": effective_lang,
                "cascade_mode": "end-to-end" if is_direct else mode,
                "x_l": r.get("x_l"),
                "x_en": None if is_direct else r.get("x_en"),
                "r_en": None if is_direct else r.get("r_en"),
                "y_en": None if is_direct else r.get("y_en"),
                "y_l": r.get("prediction"),
                "y_l_golds": gold_by_id[ex_id],
            }
        )

    if missing:
        sample = ", ".join(missing[:10])
        raise KeyError(
            f"{len(missing)} baseline rows have no gold answer match for lang={effective_lang}. "
            f"Sample example_id(s): {sample}"
        )
    return out_rows


def export_baseline_all_langs(
    pred_dir: str,
    config_path: str,
    out_prefix: Optional[str] = None,
    out_file: Optional[str] = None,
) -> str:
    cfg = load_config(config_path)
    all_rows: List[Dict[str, Any]] = []
    for lang in cfg.dataset.langs:
        effective_lang = _resolve_lang(lang, pred_dir, cfg)
        all_rows.extend(_export_one_lang(effective_lang, pred_dir, cfg))

    prefix = out_prefix or _default_prefix(cfg)
    out_name = out_file or f"{prefix}_all.json"
    out_path = os.path.join(pred_dir, out_name)
    _write_json(out_path, all_rows)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export baseline predictions with appended gold answers")
    ap.add_argument("--lang", default="", help="Language code token for input lookup/output naming")
    ap.add_argument("--all-langs", action="store_true", help="Export all dataset languages into one file")
    ap.add_argument("--pred-dir", required=True, help="Directory containing per-language prediction files")
    ap.add_argument("--config", required=True, help="V2 config used to load dataset gold answers")
    ap.add_argument(
        "--out-prefix",
        default="",
        help="Output file prefix. If omitted, derived from run_name/dataset_type.",
    )
    ap.add_argument(
        "--out-file",
        default="",
        help="Output filename for --all-langs mode (written under pred-dir).",
    )
    args = ap.parse_args()
    if args.all_langs:
        out = export_baseline_all_langs(
            pred_dir=args.pred_dir,
            config_path=args.config,
            out_prefix=args.out_prefix or None,
            out_file=args.out_file or None,
        )
    else:
        if not args.lang:
            raise ValueError("--lang is required unless --all-langs is set")
        out = export_baseline(
            lang=args.lang,
            pred_dir=args.pred_dir,
            config_path=args.config,
            out_prefix=args.out_prefix or None,
        )
    print(out)


if __name__ == "__main__":
    main()
