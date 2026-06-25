import argparse
import json
import logging
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..config import ModelEngineConfig
from ..config import V2Config, load_config
from ..loaders.registry import load_records_by_language
from . import mkqa_eval_util

LOGGER = logging.getLogger(__name__)


def _load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_prediction_rows(path: str, langs: List[str]) -> List[Dict]:
    # Backward-compatible: support either a single JSONL or a per-language directory.
    if os.path.isfile(path):
        return _load_jsonl(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    rows: List[Dict] = []
    for lang in langs:
        lang_path = os.path.join(path, f"{lang}.jsonl")
        if os.path.exists(lang_path):
            rows.extend(_load_jsonl(lang_path))
    return rows


def _write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _example_id_aliases(example_id: Any) -> List[str]:
    ex_id = str(example_id)
    aliases = [ex_id]
    if ex_id.endswith("_v1"):
        aliases.append(ex_id[: -len("_v1")])
    return aliases


def _derive_slice_from_row(r: Dict) -> str:
    # Prefer explicit slice if present.
    explicit = r.get("slice")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    error_group = str(r.get("error_group", "")).strip()
    error_type = r.get("error_type")
    cascade_mode = r.get("cascade_mode")
    if cascade_mode is None:
        cascade_mode = r.get("cascade")
    cm = str(cascade_mode or "").strip().lower()

    if error_group == "baseline":
        return f"baseline/{cm or 'unknown'}"
    if error_group == "input_err":
        return f"input_err/{error_type}/{cm or 'unknown'}"
    if error_group == "output_err":
        return f"output_err/{error_type}/{cm or 'unknown'}"
    return "all"


def _pick_binary(raw: str) -> int:
    t = (raw or "").strip()
    if not t:
        return 0
    if t in {"0", "1"}:
        return int(t)
    m = re.search(r"[01]", t)
    if m:
        return int(m.group(0))
    return 0


def _judge_prompt(question_l: str, golds: List[str], pred_0: str, pred_1: str) -> str:
    gold_text = "\n".join(f"- {g}" for g in golds[:8]) if golds else "-"
    return f"""
You are an evaluator for answer quality.

Given:
- Question:
{question_l}

- Gold answers:
{gold_text}

- Candidate 0:
{pred_0}

- Candidate 1:
{pred_1}

Return only one character:
1 if Candidate 1 is better than Candidate 0
0 otherwise

No explanation. Output strictly 0 or 1.
"""


def _compute_f1(
    groups: Dict[Tuple[str, str], List[Tuple[str, List[str]]]],
    lang: str,
) -> Dict[Tuple[str, str], float]:
    out = {}
    for key, items in groups.items():
        vals = []
        for pred, golds in items:
            if not golds:
                continue
            f1 = mkqa_eval_util.compute_max_score_over_answers(
                mkqa_eval_util.calculate_f1, pred, golds, lang
            )
            vals.append(f1)
        out[key] = round(100.0 * (sum(vals) / max(1, len(vals))), 2)
    return out


def _compute_bleu(
    groups: Dict[Tuple[str, str], List[Tuple[str, List[str]]]],
) -> Dict[Tuple[str, str], float]:
    try:
        import sacrebleu
    except ImportError as e:
        raise RuntimeError("sacrebleu is not installed") from e
    out = {}
    for key, items in groups.items():
        hyps = []
        refs_per_example = []
        for pred, golds in items:
            if not golds:
                continue
            hyps.append(pred or "")
            refs_per_example.append([g or "" for g in golds])
        if not hyps:
            out[key] = 0.0
            continue
        max_refs = max(len(r) for r in refs_per_example)
        refs_for_sacrebleu: List[List[str]] = []
        for r_idx in range(max_refs):
            refs_for_sacrebleu.append(
                [refs[r_idx] if r_idx < len(refs) else "" for refs in refs_per_example]
            )
        bleu = sacrebleu.corpus_bleu(hyps, refs_for_sacrebleu, lowercase=False, tokenize="13a")
        out[key] = round(float(bleu.score), 2)
    return out


def _compute_chrf(
    groups: Dict[Tuple[str, str], List[Tuple[str, List[str]]]],
) -> Dict[Tuple[str, str], float]:
    try:
        import sacrebleu
    except ImportError as e:
        raise RuntimeError("sacrebleu is not installed") from e

    def _normalize_chrf_text(text: str) -> str:
        return unicodedata.normalize("NFKC", text or "").lower()

    out = {}
    for key, items in groups.items():
        hyps = []
        refs_per_example = []
        for pred, golds in items:
            if not golds:
                continue
            hyps.append(_normalize_chrf_text(pred or ""))
            refs_per_example.append([_normalize_chrf_text(g or "") for g in golds])
        if not hyps:
            out[key] = 0.0
            continue
        max_refs = max(len(r) for r in refs_per_example)
        refs_for_sacrebleu: List[List[str]] = []
        for r_idx in range(max_refs):
            refs_for_sacrebleu.append(
                [refs[r_idx] if r_idx < len(refs) else "" for refs in refs_per_example]
            )
        chrf = sacrebleu.corpus_chrf(hyps, refs_for_sacrebleu)
        out[key] = round(float(chrf.score), 2)
    return out


def _compute_bertscore(
    groups: Dict[Tuple[str, str], List[Tuple[str, List[str]]]],
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
) -> Dict[Tuple[str, str], float]:
    try:
        from bert_score import score as bert_score
    except ImportError as e:
        raise RuntimeError("bert_score is not installed") from e
    out = {}
    for key, items in groups.items():
        cands = []
        refs = []
        ex_ids = []
        for idx, (pred, golds) in enumerate(items):
            if not golds:
                continue
            for g in golds:
                cands.append(pred or "")
                refs.append(g or "")
                ex_ids.append(str(idx))
        if not cands:
            out[key] = 0.0
            continue
        _p, _r, f = bert_score(
            cands,
            refs,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
            rescale_with_baseline=rescale_with_baseline,
        )
        best = {}
        for ex_id, fv in zip(ex_ids, f.detach().cpu().tolist()):
            prev = best.get(ex_id)
            if prev is None or fv > prev:
                best[ex_id] = fv
        f_avg = sum(best.values()) / max(1, len(best))
        out[key] = round(100.0 * f_avg, 2)
    return out


def _compute_win_rate_context_vs_standard(
    cfg: V2Config,
    rows: List[Dict],
    gold_by_lang: Dict[str, Dict[str, List[str]]],
    pred_field: str,
    judgment_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str], float]:
    # key: (slice_context, lang) -> list[(example_id, x_l, golds, standard_pred, context_pred)]
    pairs: Dict[Tuple[str, str], List[Tuple[str, str, List[str], str, str]]] = defaultdict(list)
    by_key: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    q_by_key: Dict[Tuple[str, str, str], str] = {}

    for r in rows:
        lang = r.get("lang")
        ex_id = str(r.get("example_id", ""))
        slice_key = _derive_slice_from_row(r)
        pred = r.get(pred_field, r.get("prediction", ""))
        if not lang or not ex_id or not slice_key:
            continue
        by_key[(lang, ex_id, slice_key)]["prediction"] = pred if isinstance(pred, str) else str(pred)
        q_by_key[(lang, ex_id, slice_key)] = r.get("x_l", "") or ""

    # Build context-vs-standard pairs.
    seen_ctx_slices = sorted({_derive_slice_from_row(r) for r in rows if _derive_slice_from_row(r).endswith("/context")})
    for ctx_slice in seen_ctx_slices:
        std_slice = ctx_slice[:-len("/context")] + "/standard"
        for lang in cfg.dataset.langs:
            recs = gold_by_lang.get(lang, {})
            for ex_id, golds in recs.items():
                ctx_pred = by_key.get((lang, str(ex_id), ctx_slice), {}).get("prediction")
                std_pred = by_key.get((lang, str(ex_id), std_slice), {}).get("prediction")
                if ctx_pred is None or std_pred is None:
                    continue
                q = q_by_key.get((lang, str(ex_id), ctx_slice), "") or q_by_key.get((lang, str(ex_id), std_slice), "")
                pairs[(ctx_slice, lang)].append((str(ex_id), q, golds, std_pred, ctx_pred))

    if not pairs:
        LOGGER.info("win_rate context_vs_standard: no comparable pairs found")
        return {}

    judge_cfg = ModelEngineConfig(
        model_type="openai",
        model_name=cfg.eval.win_judge_model_name,
        engine_kwargs={
            "api_key_env": cfg.eval.win_judge_api_key_env,
            "timeout_s": cfg.eval.win_judge_timeout_s,
        },
    )
    from ..runtime.engine_factory import create_engine

    engine = create_engine(judge_cfg)
    engine.load_model()
    try:
        out: Dict[Tuple[str, str], float] = {}
        for key, items in pairs.items():
            LOGGER.info(
                "win_rate context_vs_standard: slice=%s lang=%s pairs=%d",
                key[0],
                key[1],
                len(items),
            )
            prompts = [
                _judge_prompt(
                    question_l=q,
                    golds=golds,
                    pred_0=std_pred,
                    pred_1=ctx_pred,
                )
                for _ex_id, q, golds, std_pred, ctx_pred in items
            ]
            raw = engine.generate_batch(
                prompts,
                max_new_tokens=cfg.eval.win_max_new_tokens,
                temperature=cfg.eval.win_temperature,
                top_p=cfg.eval.win_top_p,
            )
            wins = [_pick_binary(r) for r in raw]
            out[key] = round(100.0 * (sum(wins) / max(1, len(wins))), 2)
            if judgment_rows is not None:
                for (ex_id, q, golds, std_pred, ctx_pred), raw_judge, win in zip(items, raw, wins):
                    judgment_rows.append(
                        {
                            "comparison": "context_vs_standard",
                            "slice": key[0],
                            "lang": key[1],
                            "example_id": ex_id,
                            "question": q,
                            "gold_answers": golds,
                            "candidate_0": std_pred,
                            "candidate_1": ctx_pred,
                            "judge_raw": raw_judge,
                            "judge_choice": win,
                        }
                    )
        return out
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


def _compute_win_rate_standard_vs_direct(
    cfg: V2Config,
    rows: List[Dict],
    gold_by_lang: Dict[str, Dict[str, List[str]]],
    pred_field: str,
    judgment_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str], float]:
    # Evaluate only baseline standard against baseline direct.
    std_slice = "baseline/standard"
    direct_slice = "baseline/direct"
    by_key: Dict[Tuple[str, str, str], str] = {}
    q_by_key: Dict[Tuple[str, str, str], str] = {}
    for r in rows:
        lang = r.get("lang")
        ex_id = str(r.get("example_id", ""))
        slice_key = _derive_slice_from_row(r)
        pred = r.get(pred_field, r.get("prediction", ""))
        if not lang or not ex_id or not slice_key:
            continue
        by_key[(lang, ex_id, slice_key)] = pred if isinstance(pred, str) else str(pred)
        q_by_key[(lang, ex_id, slice_key)] = r.get("x_l", "") or ""

    # If no direct rows exist, return empty map.
    has_direct = any(k[2] == direct_slice for k in by_key.keys())
    if not has_direct:
        LOGGER.info("win_rate standard_vs_direct: no baseline/direct rows found")
        return {}

    pairs: Dict[Tuple[str, str], List[Tuple[str, str, List[str], str, str]]] = defaultdict(list)
    for lang in cfg.dataset.langs:
        for ex_id, golds in gold_by_lang.get(lang, {}).items():
            std_pred = by_key.get((lang, str(ex_id), std_slice))
            direct_pred = by_key.get((lang, str(ex_id), direct_slice))
            if std_pred is None or direct_pred is None:
                continue
            q = q_by_key.get((lang, str(ex_id), std_slice), "") or q_by_key.get((lang, str(ex_id), direct_slice), "")
            pairs[(std_slice, lang)].append((str(ex_id), q, golds, direct_pred, std_pred))

    if not pairs:
        LOGGER.info("win_rate standard_vs_direct: no comparable pairs found")
        return {}

    judge_cfg = ModelEngineConfig(
        model_type="openai",
        model_name=cfg.eval.win_judge_model_name,
        engine_kwargs={
            "api_key_env": cfg.eval.win_judge_api_key_env,
            "timeout_s": cfg.eval.win_judge_timeout_s,
        },
    )
    from ..runtime.engine_factory import create_engine

    engine = create_engine(judge_cfg)
    engine.load_model()
    try:
        out: Dict[Tuple[str, str], float] = {}
        for key, items in pairs.items():
            LOGGER.info(
                "win_rate standard_vs_direct: slice=%s lang=%s pairs=%d",
                key[0],
                key[1],
                len(items),
            )
            prompts = [
                _judge_prompt(
                    question_l=q,
                    golds=golds,
                    pred_0=direct_pred,
                    pred_1=std_pred,
                )
                for _ex_id, q, golds, direct_pred, std_pred in items
            ]
            raw = engine.generate_batch(
                prompts,
                max_new_tokens=cfg.eval.win_max_new_tokens,
                temperature=cfg.eval.win_temperature,
                top_p=cfg.eval.win_top_p,
            )
            wins = [_pick_binary(r) for r in raw]
            out[key] = round(100.0 * (sum(wins) / max(1, len(wins))), 2)
            if judgment_rows is not None:
                for (ex_id, q, golds, direct_pred, std_pred), raw_judge, win in zip(items, raw, wins):
                    judgment_rows.append(
                        {
                            "comparison": "standard_vs_direct",
                            "slice": key[0],
                            "lang": key[1],
                            "example_id": ex_id,
                            "question": q,
                            "gold_answers": golds,
                            "candidate_0": direct_pred,
                            "candidate_1": std_pred,
                            "judge_raw": raw_judge,
                            "judge_choice": win,
                        }
                    )
        return out
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass


def run_metrics(
    config_path: str,
    predictions_jsonl: str = "",
    out_json: str = "",
    prediction_field: str = "",
    methods_csv: str = "",
):
    cfg: V2Config = load_config(config_path)
    pred_path = predictions_jsonl or cfg.outputs.predictions_jsonl
    out_path = out_json or cfg.outputs.metrics_json
    methods = [m.strip().lower() for m in (methods_csv or ",".join(cfg.eval.methods)).split(",") if m.strip()]
    pred_field = prediction_field or cfg.eval.prediction_field

    LOGGER.info("run_metrics start: run_name=%s dataset=%s", cfg.run_name, cfg.dataset.dataset_type)
    LOGGER.info("run_metrics inputs: predictions=%s out=%s prediction_field=%s", pred_path, out_path, pred_field)
    LOGGER.info("run_metrics methods: %s", ",".join(methods))

    rows = _load_prediction_rows(pred_path, cfg.dataset.langs)
    LOGGER.info("loaded prediction rows: %d", len(rows))
    records_by_lang = load_records_by_language(cfg.dataset)
    LOGGER.info(
        "loaded gold records by language: %s",
        {lang: len(recs) for lang, recs in records_by_lang.items()},
    )
    gold_by_lang = {}
    for lang, recs in records_by_lang.items():
        by_id = {}
        for r in recs:
            for ex_id in _example_id_aliases(r["example_id"]):
                by_id[ex_id] = r["y_l_gold"]
        gold_by_lang[lang] = by_id

    groups: Dict[Tuple[str, str], List[Tuple[str, List[str]]]] = defaultdict(list)
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rows:
        lang = r.get(cfg.eval.lang_field, r.get("lang"))
        slice_key = r.get(cfg.eval.slice_field)
        if not slice_key:
            slice_key = _derive_slice_from_row(r)
        ex_id = str(r.get(cfg.eval.example_id_field, r.get("example_id", "")))
        pred = r.get(pred_field, "")
        if not lang or not ex_id:
            continue
        golds = gold_by_lang.get(lang, {}).get(ex_id, [])
        key = (slice_key, lang)
        groups[key].append((pred if isinstance(pred, str) else str(pred), golds))
        counts[key] += 1

    LOGGER.info("grouped rows into %d (slice,lang) buckets", len(groups))

    result = {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset.dataset_type,
        "methods": methods,
        "slices": {},
    }

    by_method = {}
    if "f1" in methods:
        LOGGER.info("computing metric: f1")
        f1_scores = {}
        langs = {k[1] for k in groups.keys()}
        for lang in langs:
            lang_groups = {k: v for k, v in groups.items() if k[1] == lang}
            f1_scores.update(_compute_f1(lang_groups, lang))
        by_method["f1"] = f1_scores
    if "bleu" in methods:
        LOGGER.info("computing metric: bleu")
        by_method["bleu"] = _compute_bleu(groups)
    if "chrf" in methods:
        LOGGER.info("computing metric: chrf")
        by_method["chrf"] = _compute_chrf(groups)
    if "bertscore" in methods:
        LOGGER.info("computing metric: bertscore")
        by_method["bertscore_f1"] = _compute_bertscore(
            groups,
            model_type=cfg.eval.bertscore_model_type,
            batch_size=cfg.eval.bertscore_batch_size,
            rescale_with_baseline=cfg.eval.bertscore_rescale_with_baseline,
        )

    win_ctx = {}
    win_std = {}
    judgment_rows: Optional[List[Dict[str, Any]]] = [] if cfg.eval.win_write_judgments else None
    if "win_rate" in methods:
        LOGGER.info("computing metric: win_rate")
        win_ctx = _compute_win_rate_context_vs_standard(
            cfg,
            rows,
            gold_by_lang,
            pred_field,
            judgment_rows=judgment_rows,
        )
        win_std = _compute_win_rate_standard_vs_direct(
            cfg,
            rows,
            gold_by_lang,
            pred_field,
            judgment_rows=judgment_rows,
        )

    slice_langs: Dict[str, Dict] = defaultdict(dict)
    for (slice_key, lang), _ in groups.items():
        row = {"count": counts[(slice_key, lang)]}
        for method_name, score_map in by_method.items():
            row[method_name] = score_map.get((slice_key, lang), 0.0)
        slice_langs[slice_key][lang] = row

    for slice_key, by_lang in slice_langs.items():
        for lang, row in by_lang.items():
            ctx_key = (slice_key, lang)
            if ctx_key in win_ctx:
                row["context_vs_standard_win_rate"] = win_ctx[ctx_key]
            if ctx_key in win_std:
                row["standard_vs_direct_win_rate"] = win_std[ctx_key]
        result["slices"][slice_key] = {"by_language": by_lang}

    _write_json(out_path, result)
    LOGGER.info("run_metrics done: wrote metrics to %s", out_path)
    if judgment_rows is not None:
        judgments_path = cfg.eval.win_judgments_jsonl.strip()
        if not judgments_path:
            root, _ext = os.path.splitext(out_path)
            judgments_path = f"{root}.judge_outputs.jsonl"
        _write_jsonl(judgments_path, judgment_rows)
        LOGGER.info("run_metrics done: wrote judge outputs to %s rows=%d", judgments_path, len(judgment_rows))
    return out_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Run v2 metrics")
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--prediction_field", default="")
    ap.add_argument("--methods", default="")
    args = ap.parse_args()
    run_metrics(
        config_path=args.config,
        predictions_jsonl=args.predictions,
        out_json=args.out,
        prediction_field=args.prediction_field,
        methods_csv=args.methods,
    )


if __name__ == "__main__":
    main()
