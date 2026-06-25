import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ..config import ModelEngineConfig
from ..runtime.engine_factory import create_engine
from .runner import _judge_prompt, _pick_binary

LOGGER = logging.getLogger(__name__)


def _read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
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


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_path(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _norm_gold(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [_norm_text(x) for x in v if _norm_text(x).strip()]
    t = _norm_text(v).strip()
    return [t] if t else []


@dataclass
class JudgeConfig:
    model_name: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: int = 120
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class IOConfig:
    left_file: str = ""
    right_file: str = ""
    out_json: str = ""
    judgments_jsonl: str = ""


@dataclass
class FieldsConfig:
    id_field: str = "sample_idx"
    language_field: str = "language"
    question_field: str = "question"
    left_prediction_field: str = "mt2_result.answer"
    right_prediction_field: str = "mt2_result.answer"
    ground_truth_source: str = "left"  # left | right | file
    ground_truth_field: str = "ground_truth"
    ground_truth_file: str = ""
    ground_truth_id_field: str = "sample_idx"
    ground_truth_language_field: str = "language"
    compare_on: str = "id_and_language"  # id_only | id_and_language


@dataclass
class PairwiseWinrateConfig:
    run_name: str
    io: IOConfig
    fields: FieldsConfig = field(default_factory=FieldsConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def _load_config(path: str) -> PairwiseWinrateConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return PairwiseWinrateConfig(
        run_name=raw.get("run_name", "pairwise_winrate"),
        io=IOConfig(**raw["io"]),
        fields=FieldsConfig(**raw.get("fields", {})),
        judge=JudgeConfig(**raw.get("judge", {})),
    )


def _norm_lang(lang: Any) -> str:
    return _norm_text(lang).strip()


def _build_key(ex_id: str, lang: str, compare_on: str) -> Tuple[str, ...]:
    if compare_on == "id_only":
        return (ex_id,)
    return (ex_id, lang)


def run_pairwise_winrate(config_path: str) -> str:
    cfg = _load_config(config_path)
    LOGGER.info("pairwise winrate start: run_name=%s", cfg.run_name)
    LOGGER.info("left=%s right=%s out=%s", cfg.io.left_file, cfg.io.right_file, cfg.io.out_json)

    left_rows = _read_rows(cfg.io.left_file)
    right_rows = _read_rows(cfg.io.right_file)
    LOGGER.info("rows: left=%d right=%d", len(left_rows), len(right_rows))

    left_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for r in left_rows:
        ex_id = _norm_text(_get_path(r, cfg.fields.id_field)).strip()
        lang = _norm_lang(_get_path(r, cfg.fields.language_field))
        if not ex_id:
            continue
        key = _build_key(ex_id, lang, cfg.fields.compare_on)
        left_map[key] = {
            "id": ex_id,
            "lang": lang,
            "pred": _norm_text(_get_path(r, cfg.fields.left_prediction_field)).strip(),
            "question": _norm_text(_get_path(r, cfg.fields.question_field)).strip(),
            "ground_truth": _norm_gold(_get_path(r, cfg.fields.ground_truth_field)),
        }

    right_map: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for r in right_rows:
        ex_id = _norm_text(_get_path(r, cfg.fields.id_field)).strip()
        lang = _norm_lang(_get_path(r, cfg.fields.language_field))
        if not ex_id:
            continue
        key = _build_key(ex_id, lang, cfg.fields.compare_on)
        right_map[key] = {
            "id": ex_id,
            "lang": lang,
            "pred": _norm_text(_get_path(r, cfg.fields.right_prediction_field)).strip(),
            "question": _norm_text(_get_path(r, cfg.fields.question_field)).strip(),
            "ground_truth": _norm_gold(_get_path(r, cfg.fields.ground_truth_field)),
        }

    gold_map: Dict[Tuple[str, ...], List[str]] = {}
    if cfg.fields.ground_truth_source == "file":
        if not cfg.fields.ground_truth_file:
            raise ValueError("fields.ground_truth_file is required when ground_truth_source=file")
        gold_rows = _read_rows(cfg.fields.ground_truth_file)
        for r in gold_rows:
            ex_id = _norm_text(_get_path(r, cfg.fields.ground_truth_id_field)).strip()
            lang = _norm_lang(_get_path(r, cfg.fields.ground_truth_language_field))
            if not ex_id:
                continue
            key = _build_key(ex_id, lang, cfg.fields.compare_on)
            gold_map[key] = _norm_gold(_get_path(r, cfg.fields.ground_truth_field))
    else:
        source = left_map if cfg.fields.ground_truth_source == "left" else right_map
        for k, v in source.items():
            gold_map[k] = v.get("ground_truth", [])

    keys = sorted(set(left_map.keys()) & set(right_map.keys()) & set(gold_map.keys()))
    LOGGER.info("comparable keys=%d", len(keys))

    grouped: Dict[str, List[Tuple[str, str, List[str], str, str]]] = defaultdict(list)
    for k in keys:
        l = left_map[k]
        r = right_map[k]
        gold = gold_map.get(k, [])
        if not gold:
            continue
        lang = r["lang"] or l["lang"]
        question = l["question"] or r["question"]
        grouped[lang].append((l["id"], question, gold, l["pred"], r["pred"]))

    judge_cfg = ModelEngineConfig(
        model_type="openai",
        model_name=cfg.judge.model_name,
        engine_kwargs={"api_key_env": cfg.judge.api_key_env, "timeout_s": cfg.judge.timeout_s},
    )
    engine = create_engine(judge_cfg)
    judgments: List[Dict[str, Any]] = []
    by_lang: Dict[str, Dict[str, Any]] = {}
    engine.load_model()
    try:
        for lang, items in sorted(grouped.items()):
            LOGGER.info("lang=%s pairs=%d", lang, len(items))
            prompts = [_judge_prompt(q, gold, pred_l, pred_r) for _id, q, gold, pred_l, pred_r in items]
            raw = engine.generate_batch(
                prompts,
                max_new_tokens=cfg.judge.max_new_tokens,
                temperature=cfg.judge.temperature,
                top_p=cfg.judge.top_p,
            )
            wins = [_pick_binary(x) for x in raw]
            for (ex_id, q, gold, pred_l, pred_r), raw_j, win in zip(items, raw, wins):
                judgments.append(
                    {
                        "run_name": cfg.run_name,
                        "lang": lang,
                        "example_id": ex_id,
                        "question": q,
                        "gold_answers": gold,
                        "candidate_0": pred_l,
                        "candidate_1": pred_r,
                        "judge_raw": raw_j,
                        "judge_choice": win,
                    }
                )
            by_lang[lang] = {
                "count": len(items),
                "win_rate_candidate_1_over_0": round(100.0 * (sum(wins) / max(1, len(wins))), 2),
            }
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass

    total_n = sum(v["count"] for v in by_lang.values())
    total_w = 0.0
    for lang, v in by_lang.items():
        total_w += (v["win_rate_candidate_1_over_0"] / 100.0) * v["count"]
    overall = round(100.0 * (total_w / max(1, total_n)), 2)

    out = {
        "run_name": cfg.run_name,
        "left_file": cfg.io.left_file,
        "right_file": cfg.io.right_file,
        "left_prediction_field": cfg.fields.left_prediction_field,
        "right_prediction_field": cfg.fields.right_prediction_field,
        "ground_truth_source": cfg.fields.ground_truth_source,
        "ground_truth_field": cfg.fields.ground_truth_field,
        "compare_on": cfg.fields.compare_on,
        "count": total_n,
        "overall_win_rate_candidate_1_over_0": overall,
        "by_language": by_lang,
    }
    _write_json(cfg.io.out_json, out)
    if cfg.io.judgments_jsonl:
        _write_jsonl(cfg.io.judgments_jsonl, judgments)
        LOGGER.info("wrote judgments jsonl: %s rows=%d", cfg.io.judgments_jsonl, len(judgments))
    LOGGER.info("pairwise winrate done: %s", cfg.io.out_json)
    return cfg.io.out_json


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Compute language-level pairwise win-rate between two files")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_pairwise_winrate(args.config)


if __name__ == "__main__":
    main()
