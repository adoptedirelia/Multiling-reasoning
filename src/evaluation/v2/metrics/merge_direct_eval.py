import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_base_lang(path: str, lang: str) -> List[Dict]:
    p = os.path.join(path, f"{lang}.jsonl")
    if not os.path.exists(p):
        return []
    return _load_jsonl(p)


def _load_direct_lang(path: str, lang: str) -> List[Dict]:
    p = os.path.join(path, f"{lang}.jsonl")
    if not os.path.exists(p):
        return []
    return _load_jsonl(p)


def _has_direct_slice(row: Dict) -> bool:
    s = str(row.get("slice", "")).strip().lower()
    if s == "baseline/direct":
        return True
    eg = str(row.get("error_group", "")).strip().lower()
    cm = str(row.get("cascade_mode", row.get("cascade", ""))).strip().lower()
    return eg == "baseline" and cm == "direct"


def merge_direct_eval(
    base_dir: str,
    direct_dir: str,
    out_dir: str,
    langs: List[str],
) -> Tuple[int, int]:
    LOGGER.info("merge_direct_eval start base_dir=%s direct_dir=%s out_dir=%s langs=%s", base_dir, direct_dir, out_dir, langs)
    os.makedirs(out_dir, exist_ok=True)
    total_added = 0
    total_skipped = 0

    for lang in langs:
        base_rows = _load_base_lang(base_dir, lang)
        direct_rows = _load_direct_lang(direct_dir, lang)
        if not base_rows:
            LOGGER.info("lang=%s skipped: missing base rows", lang)
            continue
        LOGGER.info("lang=%s base_rows=%d direct_rows=%d", lang, len(base_rows), len(direct_rows))

        q_by_ex: Dict[str, str] = {}
        for r in base_rows:
            ex_id = str(r.get("example_id", ""))
            if not ex_id:
                continue
            x_l = r.get("x_l")
            if isinstance(x_l, str) and x_l.strip() and ex_id not in q_by_ex:
                q_by_ex[ex_id] = x_l

        existing_direct = {
            str(r.get("example_id", ""))
            for r in base_rows
            if _has_direct_slice(r)
        }

        merged = list(base_rows)
        for dr in direct_rows:
            ex_id = str(dr.get("example_id", ""))
            if not ex_id:
                total_skipped += 1
                continue
            if ex_id in existing_direct:
                total_skipped += 1
                continue
            pred = dr.get("prediction")
            if not isinstance(pred, str) or not pred.strip():
                total_skipped += 1
                continue

            merged.append(
                {
                    "run_name": str(base_rows[0].get("run_name", "")),
                    "lang": lang,
                    "example_id": ex_id,
                    "error_group": "baseline",
                    "error_type": None,
                    "cascade": "direct",
                    "cascade_mode": "direct",
                    "slice": "baseline/direct",
                    "x_l": q_by_ex.get(ex_id, ""),
                    "x_en": None,
                    "r_en": None,
                    "y_en": None,
                    "y_l": pred,
                    "prediction": pred,
                    "x_en_err": None,
                    "r_en_err": None,
                    "y_en_err": None,
                }
            )
            total_added += 1

        _write_jsonl(os.path.join(out_dir, f"{lang}.jsonl"), merged)
        LOGGER.info("lang=%s merged_rows=%d", lang, len(merged))

    LOGGER.info("merge_direct_eval done added=%d skipped=%d out_dir=%s", total_added, total_skipped, out_dir)
    return total_added, total_skipped


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Merge direct-eval predictions into cascade prediction files")
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--direct_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--langs", required=True, help="Comma-separated language list")
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    added, skipped = merge_direct_eval(
        base_dir=args.base_dir,
        direct_dir=args.direct_dir,
        out_dir=args.out_dir,
        langs=langs,
    )
    print(f"Merged direct-eval rows. added={added} skipped={skipped} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
