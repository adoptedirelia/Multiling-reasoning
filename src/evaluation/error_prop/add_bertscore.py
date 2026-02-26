import argparse
import json
import os
from typing import Dict, List, Tuple

from .config import load_config
from .dataset_loader import load_records_for_language
from .direct_config import load_direct_config
from .mkqa_loader import load_mkqa_records_for_ids


def _load_predictions(path: str) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex_id = str(row.get("example_id", ""))
            if not ex_id:
                continue
            pred = row.get("prediction", "")
            preds[ex_id] = pred if isinstance(pred, str) else str(pred)
    return preds


def _load_gold_map(cfg, lang: str, example_ids: List[str]) -> Dict[str, List[str]]:
    if cfg.dataset.dataset_type == "mkqa":
        recs = load_mkqa_records_for_ids(cfg.dataset.mkqa_path, lang, example_ids)
        return {ex_id: rec.get("y_l_gold", []) for ex_id, rec in recs.items()}

    # For non-MKQA datasets, load full language split once and filter by ID.
    records = load_records_for_language(
        cfg.dataset,
        lang,
        max_examples_override=10**9,
    )
    rec_map = {str(r.get("example_id", "")): r for r in records}
    return {ex_id: rec_map.get(ex_id, {}).get("y_l_gold", []) for ex_id in example_ids}


def _build_pairwise_examples(
    preds: Dict[str, str],
    golds: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    cand_texts: List[str] = []
    ref_texts: List[str] = []
    ex_ids: List[str] = []
    for ex_id, pred in preds.items():
        refs = golds.get(ex_id, [])
        if not refs:
            continue
        for ref in refs:
            cand_texts.append(pred or "")
            ref_texts.append(ref or "")
            ex_ids.append(ex_id)
    return cand_texts, ref_texts, ex_ids


def _reduce_by_example(
    ex_ids: List[str],
    p_vals: List[float],
    r_vals: List[float],
    f_vals: List[float],
) -> Tuple[float, float, float, int]:
    best: Dict[str, Tuple[float, float, float]] = {}
    for ex_id, p, r, f in zip(ex_ids, p_vals, r_vals, f_vals):
        prev = best.get(ex_id)
        if prev is None or f > prev[2]:
            best[ex_id] = (p, r, f)
    if not best:
        return 0.0, 0.0, 0.0, 0
    p_avg = sum(v[0] for v in best.values()) / len(best)
    r_avg = sum(v[1] for v in best.values()) / len(best)
    f_avg = sum(v[2] for v in best.values()) / len(best)
    return p_avg, r_avg, f_avg, len(best)


def _compute_bertscore(
    cand_texts: List[str],
    ref_texts: List[str],
    ex_ids: List[str],
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
) -> Dict[str, float]:
    try:
        from bert_score import score as bert_score
    except ImportError as e:
        raise RuntimeError(
            "bert_score is not installed in this environment. "
            "Install it in faiss_build_env first."
        ) from e

    if not cand_texts:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
            "bertscore_num_examples": 0,
        }

    p, r, f = bert_score(
        cand_texts,
        ref_texts,
        model_type=model_type,
        batch_size=batch_size,
        verbose=False,
        rescale_with_baseline=rescale_with_baseline,
    )
    p_avg, r_avg, f_avg, n = _reduce_by_example(
        ex_ids,
        p.detach().cpu().tolist(),
        r.detach().cpu().tolist(),
        f.detach().cpu().tolist(),
    )
    return {
        "bertscore_precision": round(100.0 * p_avg, 2),
        "bertscore_recall": round(100.0 * r_avg, 2),
        "bertscore_f1": round(100.0 * f_avg, 2),
        "bertscore_num_examples": n,
    }


def _bleu_tokenizer_for_lang(lang: str) -> str:
    l = (lang or "").lower()
    if l in {"zh", "zh_cn", "zho"}:
        return "zh"
    return "13a"


def _compute_bleu(
    preds: Dict[str, str],
    golds: Dict[str, List[str]],
    lang: str,
) -> Dict[str, float]:
    try:
        import sacrebleu
    except ImportError as e:
        raise RuntimeError(
            "sacrebleu is not installed in this environment. "
            "Install it in faiss_build_env first."
        ) from e

    hyps: List[str] = []
    refs_per_example: List[List[str]] = []
    for ex_id, pred in preds.items():
        refs = golds.get(ex_id, [])
        if not refs:
            continue
        hyps.append(pred or "")
        refs_per_example.append([r or "" for r in refs])

    if not hyps:
        return {"bleu": 0.0, "bleu_num_examples": 0}

    max_refs = max(len(r) for r in refs_per_example)
    refs_for_sacrebleu: List[List[str]] = []
    for r_idx in range(max_refs):
        refs_for_sacrebleu.append(
            [
                refs[r_idx] if r_idx < len(refs) else ""
                for refs in refs_per_example
            ]
        )

    bleu = sacrebleu.corpus_bleu(
        hyps,
        refs_for_sacrebleu,
        lowercase=False,
        tokenize=_bleu_tokenizer_for_lang(lang),
    )
    return {
        "bleu": round(float(bleu.score), 2),
        "bleu_num_examples": len(hyps),
    }


def _iter_prediction_files(predictions_dir: str):
    for dirpath, _dirnames, filenames in os.walk(predictions_dir):
        for name in filenames:
            if not name.endswith(".jsonl"):
                continue
            pred_path = os.path.join(dirpath, name)
            rel = os.path.relpath(pred_path, predictions_dir)
            parts = rel.split(os.sep)
            if len(parts) < 2:
                continue
            lang = os.path.splitext(parts[-1])[0]
            tag = "/".join(parts[:-1])
            yield tag, lang, pred_path


def _load_any_config(config_path: str):
    try:
        return load_config(config_path)
    except Exception as err_error_prop:
        try:
            return load_direct_config(config_path)
        except Exception as err_direct:
            raise RuntimeError(
                "Failed to parse config as error_prop or direct_eval.\n"
                f"error_prop parse error: {err_error_prop}\n"
                f"direct_eval parse error: {err_direct}"
            ) from err_direct


def run(
    config_path: str,
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
    add_bertscore: bool,
    add_bleu: bool,
):
    if not add_bertscore and not add_bleu:
        raise ValueError("At least one metric must be enabled (--add_bertscore and/or --add_bleu).")

    cfg = _load_any_config(config_path)
    pred_root = cfg.outputs.predictions_dir
    metrics_root = cfg.outputs.metrics_dir

    gold_cache: Dict[str, Dict[str, List[str]]] = {}
    updated = 0
    skipped = 0

    for tag, lang, pred_path in _iter_prediction_files(pred_root):
        metrics_path = os.path.join(metrics_root, tag, lang, "metrics.json")
        if not os.path.exists(metrics_path):
            skipped += 1
            continue

        preds = _load_predictions(pred_path)
        ex_ids = list(preds.keys())
        if not ex_ids:
            skipped += 1
            continue

        if lang not in gold_cache:
            gold_cache[lang] = _load_gold_map(cfg, lang, ex_ids)
        else:
            # Fill any new IDs not seen in previous tags for this language.
            missing = [ex_id for ex_id in ex_ids if ex_id not in gold_cache[lang]]
            if missing:
                extra = _load_gold_map(cfg, lang, missing)
                gold_cache[lang].update(extra)

        golds = {ex_id: gold_cache[lang].get(ex_id, []) for ex_id in ex_ids}
        updates = {}
        if add_bertscore:
            cand_texts, ref_texts, pair_ex_ids = _build_pairwise_examples(preds, golds)
            bs = _compute_bertscore(
                cand_texts,
                ref_texts,
                pair_ex_ids,
                model_type=model_type,
                batch_size=batch_size,
                rescale_with_baseline=rescale_with_baseline,
            )
            updates.update(
                {
                    "bertscore_model_type": model_type,
                    "bertscore_rescale_with_baseline": rescale_with_baseline,
                    **bs,
                }
            )

        if add_bleu:
            updates.update(_compute_bleu(preds, golds, lang))

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        metrics.update(updates)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        updated += 1

    print(
        "Updated metrics files with "
        f"{'BERTScore' if add_bertscore else ''}"
        f"{' + ' if add_bertscore and add_bleu else ''}"
        f"{'BLEU' if add_bleu else ''}: {updated}; "
        f"skipped (missing metrics/preds): {skipped}"
    )


def main():
    ap = argparse.ArgumentParser(description="Add BERTScore and/or BLEU to existing metrics")
    ap.add_argument("--config", required=True, help="Error-prop config used for the run")
    ap.add_argument(
        "--model_type",
        default="xlm-roberta-large",
        help="BERTScore model_type (default: xlm-roberta-large)",
    )
    ap.add_argument("--batch_size", type=int, default=16, help="BERTScore batch size")
    ap.add_argument(
        "--rescale_with_baseline",
        action="store_true",
        help="Use BERTScore baseline rescaling",
    )
    ap.add_argument(
        "--add_bertscore",
        action="store_true",
        default=True,
        help="Add BERTScore fields (default: enabled)",
    )
    ap.add_argument(
        "--no_bertscore",
        action="store_false",
        dest="add_bertscore",
        help="Disable BERTScore",
    )
    ap.add_argument(
        "--add_bleu",
        action="store_true",
        default=True,
        help="Add BLEU fields (default: enabled)",
    )
    ap.add_argument(
        "--no_bleu",
        action="store_false",
        dest="add_bleu",
        help="Disable BLEU",
    )
    args = ap.parse_args()
    run(
        config_path=args.config,
        model_type=args.model_type,
        batch_size=args.batch_size,
        rescale_with_baseline=args.rescale_with_baseline,
        add_bertscore=args.add_bertscore,
        add_bleu=args.add_bleu,
    )


if __name__ == "__main__":
    main()
