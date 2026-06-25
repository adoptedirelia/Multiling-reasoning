"""
GPT-based evaluation pipeline (end_to_end / cascade / prompting).
Imports only OpenAIEngine — no vllm/torch loaded at startup.
"""
import os
import re
import json
import argparse
from typing import Dict, List, Optional

# Import engine directly to bypass __init__.py (which imports vllm)
from src.eval.engine.openai import OpenAIEngine
from src.eval.config import EvalConfig, ModelConfig, load_config_from_dict
from src.prompt import (
    MT1_PROMPT, MT2_PROMPT, REASONING_PROMPT, MT2_BASE_PROMPT, END_TO_END_PROMPT,
    MATH_REASONING_PROMPT, MATH_MT2_BASE_PROMPT, MATH_MT2_PROMPT, MATH_END_TO_END_PROMPT,
)
from src.prompt_mc import (
    MC_MT1_PROMPT, MC_MT2_PROMPT, MC_REASONING_PROMPT,
    MC_MT2_BASE_PROMPT, MC_END_TO_END_PROMPT,
    build_options_text, build_answer_hint, build_options_translation_format,
    OPTION_FIELDS,
)

_OPTION_FIELDS = ["option_a", "option_b", "option_c", "option_d", "option_e"]
_ENGLISH_OPTION_FIELDS = ["english_option_a", "english_option_b", "english_option_c", "english_option_d", "english_option_e"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mc_format_kwargs(opts: Dict[str, str]) -> Dict[str, str]:
    return {
        "options_text": build_options_text(opts),
        "answer_hint": build_answer_hint(opts),
        "options_translation_format": build_options_translation_format(opts),
    }


def _extract_options(dataset):
    if not dataset or "option_a" not in dataset[0]:
        return None
    return [{f: s.get(f, "") for f in _OPTION_FIELDS} for s in dataset]


def _extract_english_options(dataset):
    if not dataset or "english_option_a" not in dataset[0]:
        return None
    return [
        {f: item.get(ef, "") for f, ef in zip(_OPTION_FIELDS, _ENGLISH_OPTION_FIELDS)}
        for item in dataset
    ]


def _ckpt_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.ckpt{ext}"


def sample_per_language(dataset: list, n: int, seed: int = 42) -> list:
    """Return at most n samples per language, in a stable order (sorted by original index)."""
    import random
    rng = random.Random(seed)
    by_lang: Dict[str, list] = {}
    for i, item in enumerate(dataset):
        lang = item.get("language", "unknown")
        by_lang.setdefault(lang, []).append((i, item))
    sampled = []
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        chosen = rng.sample(pool, min(n, len(pool)))
        sampled.extend(chosen)
    sampled.sort(key=lambda x: x[0])
    return [item for _, item in sampled]


def load_json_dataset(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        result = []
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item["language"] = key
                    result.append(item)
        return result
    return data


def save_results(results, output_path: str):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _load_checkpoint(path: str) -> list:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data
    return []


def _save_checkpoint(results: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

def extract_translation_output(text: str) -> Dict[str, str]:
    result = {"translation": ""}
    m = re.search(r'<question_translation>(.*?)</question_translation>', text, re.DOTALL)
    if m:
        result["translation"] = m.group(1).strip()
    else:
        m = re.search(r'<translation>(.*?)</translation>', text, re.DOTALL)
        if m:
            result["translation"] = m.group(1).strip()
    for field in OPTION_FIELDS:
        m = re.search(rf'<{field}_translation>(.*?)</{field}_translation>', text, re.DOTALL)
        if m:
            result[field] = m.group(1).strip()
    return result


def extract_reasoning_output(text: str) -> Dict[str, str]:
    result = {"reasoning": "", "answer": ""}
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        result["answer"] = m.group(1).strip()
    return result


def extract_math_output(text: str) -> Dict[str, str]:
    result = {"reasoning": "", "answer": ""}
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()
    boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    if boxed_matches:
        result["answer"] = boxed_matches[-1].strip()
    else:
        m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if m:
            result["answer"] = m.group(1).strip()
    return result


def extract_mt2_output(text: str) -> Dict[str, str]:
    result = {"reasoning": "", "answer": ""}
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if m:
        result["reasoning"] = m.group(1).strip()
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        result["answer"] = m.group(1).strip()
    return result


# ---------------------------------------------------------------------------
# Engine factory (OpenAI only)
# ---------------------------------------------------------------------------

def create_openai_engine(model_config: ModelConfig) -> OpenAIEngine:
    return OpenAIEngine(
        model_name=model_config.model_name,
        api_key_env=model_config.api_key_env,
        base_url=model_config.base_url,
        timeout_s=model_config.timeout_s,
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_translation_batch(engine: OpenAIEngine, questions: List[str],
                          model_config: ModelConfig,
                          options: Optional[List[Dict]] = None) -> List[Dict]:
    if options:
        prompts = [MC_MT1_PROMPT.format(question=q, **_mc_format_kwargs(o))
                   for q, o in zip(questions, options)]
    else:
        prompts = [MT1_PROMPT.format(question=q) for q in questions]
    outputs = engine.generate_batch(prompts,
                                    max_new_tokens=model_config.max_new_tokens,
                                    temperature=model_config.temperature,
                                    top_p=model_config.top_p)
    return [extract_translation_output(o) for o in outputs]


def run_reasoning(engine: OpenAIEngine, questions: List[str],
                  model_config: ModelConfig,
                  options: Optional[List[Dict]] = None,
                  is_math: bool = False) -> List[Dict]:
    if options:
        prompts = [MC_REASONING_PROMPT.format(question=q, **_mc_format_kwargs(o))
                   for q, o in zip(questions, options)]
    elif is_math:
        prompts = [MATH_REASONING_PROMPT.format(question=q) for q in questions]
    else:
        prompts = [REASONING_PROMPT.format(question=q) for q in questions]
    outputs = engine.generate_batch(prompts,
                                    max_new_tokens=model_config.max_new_tokens,
                                    temperature=model_config.temperature,
                                    top_p=model_config.top_p)
    extractor = extract_math_output if is_math else extract_reasoning_output
    return [extractor(o) for o in outputs]


def run_mt2_base(engine: OpenAIEngine, questions: List[str],
                 english_answers: List[str], languages: List[str],
                 model_config: ModelConfig,
                 options: Optional[List[Dict]] = None,
                 is_math: bool = False) -> List[Dict]:
    prompts = []
    for i in range(len(questions)):
        if options:
            prompts.append(MC_MT2_BASE_PROMPT.format(
                English_Answer=english_answers[i],
                language=languages[i],
                **_mc_format_kwargs(options[i]),
            ))
        elif is_math:
            prompts.append(MATH_MT2_BASE_PROMPT.format(
                English_Answer=english_answers[i],
                language=languages[i],
            ))
        else:
            prompts.append(MT2_BASE_PROMPT.format(
                English_Answer=english_answers[i],
                language=languages[i],
            ))
    outputs = engine.generate_batch(prompts,
                                    max_new_tokens=model_config.max_new_tokens,
                                    temperature=model_config.temperature,
                                    top_p=model_config.top_p)
    extractor = extract_math_output if is_math else extract_mt2_output
    return [extractor(o) for o in outputs]


def run_mt2(engine: OpenAIEngine, questions: List[str],
            english_questions: List[str],
            english_thinking_processes: List[str],
            english_answers: List[str],
            languages: List[str],
            model_config: ModelConfig,
            options: Optional[List[Dict]] = None,
            is_math: bool = False) -> List[Dict]:
    prompts = []
    for i in range(len(questions)):
        if options:
            prompts.append(MC_MT2_PROMPT.format(
                question=questions[i],
                language=languages[i],
                English_Question=english_questions[i],
                English_Thinking_Process=english_thinking_processes[i],
                English_Answer=english_answers[i],
                **_mc_format_kwargs(options[i]),
            ))
        elif is_math:
            prompts.append(MATH_MT2_PROMPT.format(
                question=questions[i],
                language=languages[i],
                English_Question=english_questions[i],
                English_Thinking_Process=english_thinking_processes[i],
                English_Answer=english_answers[i],
            ))
        else:
            prompts.append(MT2_PROMPT.format(
                question=questions[i],
                language=languages[i],
                English_Question=english_questions[i],
                English_Thinking_Process=english_thinking_processes[i],
                English_Answer=english_answers[i],
            ))
    outputs = engine.generate_batch(prompts,
                                    max_new_tokens=model_config.max_new_tokens,
                                    temperature=model_config.temperature,
                                    top_p=model_config.top_p)
    extractor = extract_math_output if is_math else extract_mt2_output
    return [extractor(o) for o in outputs]


def run_end_to_end(engine: OpenAIEngine, questions: List[str],
                   languages: List[str], model_config: ModelConfig,
                   options: Optional[List[Dict]] = None,
                   is_math: bool = False) -> List[Dict]:
    prompts = []
    for i, q in enumerate(questions):
        lang = languages[i] if languages else ""
        if options:
            prompts.append(MC_END_TO_END_PROMPT.format(
                question=q, language=lang, **_mc_format_kwargs(options[i])))
        elif is_math:
            prompts.append(MATH_END_TO_END_PROMPT.format(question=q, language=lang))
        else:
            prompts.append(END_TO_END_PROMPT.format(question=q, language=lang))
    outputs = engine.generate_batch(prompts,
                                    max_new_tokens=model_config.max_new_tokens,
                                    temperature=model_config.temperature,
                                    top_p=model_config.top_p)
    extractor = extract_math_output if is_math else extract_mt2_output
    return [extractor(o) for o in outputs]


# ---------------------------------------------------------------------------
# MT1 translation with checkpointing
# ---------------------------------------------------------------------------

def translate_questions(config: EvalConfig, dataset: list) -> list:
    all_questions = [s.get("question", "") for s in dataset]
    all_languages = [s.get("language", "") for s in dataset]
    all_answers = [s.get("answer", "") for s in dataset]
    all_options = _extract_options(dataset)
    is_mc = all_options is not None

    translation_path = os.path.join(config.output_dir, config.translation_file)
    ckpt_path = _ckpt_path(translation_path)

    ckpt_data = _load_checkpoint(ckpt_path)
    start_idx = len(ckpt_data)
    if start_idx >= len(dataset):
        print(f"MT1 already complete ({len(ckpt_data)}/{len(dataset)} samples) — skipping translation")
        save_results(ckpt_data, translation_path)
        return ckpt_data
    if start_idx:
        print(f"Resuming MT1 from checkpoint: {start_idx}/{len(dataset)}")

    if start_idx < len(dataset):
        engine = create_openai_engine(config.mt1_config)
        engine.load_model()

        for i in range(start_idx, len(all_questions), config.batch_size):
            batch_end = min(i + config.batch_size, len(all_questions))
            batch_q = all_questions[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None

            batch_out = run_translation_batch(engine, batch_q, config.mt1_config, batch_opts)

            for j, out in enumerate(batch_out):
                idx = i + j
                entry = {
                    "sample_idx": idx,
                    "question": all_questions[idx],
                    "english_question": out.get("translation", ""),
                    "language": all_languages[idx],
                    "answer": all_answers[idx],
                }
                if is_mc:
                    entry.update(all_options[idx])
                    for field in _OPTION_FIELDS:
                        entry[f"english_{field}"] = out.get(field, "")
                ckpt_data.append(entry)

            _save_checkpoint(ckpt_data, ckpt_path)
            print(f"  MT1 checkpoint: {len(ckpt_data)}/{len(dataset)}")

    print(f"Translation done: {len(ckpt_data)} samples")
    save_results(ckpt_data, translation_path)
    return ckpt_data


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(config: EvalConfig, dataset: list):
    if config.num_samples:
        dataset = dataset[:config.num_samples]

    all_questions = [s.get("question", "") for s in dataset]
    all_languages = [s.get("language", "") for s in dataset]
    all_ground_truths = [s.get("answer", "") for s in dataset]
    english_questions = [s.get("english_question", "") for s in dataset]
    all_options = _extract_options(dataset)
    english_options = _extract_english_options(dataset)
    is_math = getattr(config, "question_type", "auto") == "math"

    baseline_type = config.baseline_type.lower()
    output_path = os.path.join(config.output_dir, config.output_file)
    results = _load_checkpoint(output_path)
    start_idx = len(results)
    if start_idx >= len(dataset):
        print(f"Already complete ({len(results)}/{len(dataset)} samples) — skipping: {output_path}")
        return
    if start_idx:
        print(f"Resuming from checkpoint: {start_idx}/{len(dataset)}")

    batch_size = config.batch_size

    if baseline_type == "end_to_end":
        engine = create_openai_engine(config.mt1_config)
        engine.load_model()
        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_q = all_questions[i:batch_end]
            batch_lang = all_languages[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            batch_out = run_end_to_end(engine, batch_q, batch_lang, config.mt1_config, batch_opts, is_math=is_math)
            for j, out in enumerate(batch_out):
                results.append({
                    "sample_idx": i + j,
                    "question": batch_q[j],
                    "language": batch_lang[j],
                    "ground_truth": all_ground_truths[i + j],
                    "result": out,
                })
            _save_checkpoint(results, output_path)
            print(f"  end_to_end checkpoint: {len(results)}/{len(dataset)}")

    elif baseline_type == "cascade":
        intermediate_path = os.path.join(config.output_dir, config.intermediate_file)
        llm_results = _load_checkpoint(intermediate_path)
        start_llm = len(llm_results)

        if start_llm < len(dataset):
            llm_engine = create_openai_engine(config.llm_config)
            llm_engine.load_model()
            for i in range(start_llm, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_eq = english_questions[i:batch_end]
                batch_opts = english_options[i:batch_end] if english_options else None
                batch_out = run_reasoning(llm_engine, batch_eq, config.llm_config, batch_opts, is_math=is_math)
                llm_results.extend(batch_out)
                _save_checkpoint(llm_results, intermediate_path)
                print(f"  cascade LLM checkpoint: {len(llm_results)}/{len(dataset)}")
            print(f"Reasoning done: {len(llm_results)} samples")

        mt2_engine = create_openai_engine(config.mt2_config)
        mt2_engine.load_model()
        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_q = all_questions[i:batch_end]
            batch_eq = english_questions[i:batch_end]
            batch_lang = all_languages[i:batch_end]
            batch_llm = llm_results[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            batch_out = run_mt2_base(
                mt2_engine, batch_q,
                english_answers=[r.get("answer", "") for r in batch_llm],
                languages=batch_lang,
                model_config=config.mt2_config,
                options=batch_opts,
                is_math=is_math,
            )
            for j in range(len(batch_q)):
                results.append({
                    "sample_idx": i + j,
                    "question": batch_q[j],
                    "language": batch_lang[j],
                    "ground_truth": all_ground_truths[i + j],
                    "english_question": batch_eq[j],
                    "llm_result": batch_llm[j],
                    "mt2_result": batch_out[j],
                })
            _save_checkpoint(results, output_path)
            print(f"  cascade MT2 checkpoint: {len(results)}/{len(dataset)}")

    elif baseline_type == "prompting":
        intermediate_path = os.path.join(config.output_dir, config.intermediate_file)
        llm_results = _load_checkpoint(intermediate_path)
        start_llm = len(llm_results)

        if start_llm < len(dataset):
            llm_engine = create_openai_engine(config.llm_config)
            llm_engine.load_model()
            for i in range(start_llm, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_eq = english_questions[i:batch_end]
                batch_opts = english_options[i:batch_end] if english_options else None
                batch_out = run_reasoning(llm_engine, batch_eq, config.llm_config, batch_opts, is_math=is_math)
                llm_results.extend(batch_out)
                _save_checkpoint(llm_results, intermediate_path)
                print(f"  prompting LLM checkpoint: {len(llm_results)}/{len(dataset)}")
            print(f"Reasoning done: {len(llm_results)} samples")

        mt2_engine = create_openai_engine(config.mt2_config)
        mt2_engine.load_model()
        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_q = all_questions[i:batch_end]
            batch_eq = english_questions[i:batch_end]
            batch_lang = all_languages[i:batch_end]
            batch_llm = llm_results[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            batch_out = run_mt2(
                mt2_engine, batch_q, batch_eq,
                english_thinking_processes=[r.get("reasoning", "") for r in batch_llm],
                english_answers=[r.get("answer", "") for r in batch_llm],
                languages=batch_lang,
                model_config=config.mt2_config,
                options=batch_opts,
                is_math=is_math,
            )
            for j in range(len(batch_q)):
                results.append({
                    "sample_idx": i + j,
                    "question": batch_q[j],
                    "language": batch_lang[j],
                    "ground_truth": all_ground_truths[i + j],
                    "english_question": batch_eq[j],
                    "llm_result": batch_llm[j],
                    "mt2_result": batch_out[j],
                })
            _save_checkpoint(results, output_path)
            print(f"  prompting MT2 checkpoint: {len(results)}/{len(dataset)}")

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    print(f"\nDone. {len(results)} samples -> {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPT-based pipeline evaluation (no vllm)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", help="Override dataset_path in config")
    parser.add_argument("--output_dir", help="Override output_dir in config")
    parser.add_argument("--baseline_type", choices=["end_to_end", "cascade", "prompting"])
    parser.add_argument("--output_file")
    parser.add_argument("--translation_file")
    parser.add_argument("--intermediate_file")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--num_samples_per_lang", type=int,
                        help="Sample at most this many items per language (random, seed=42)")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for per-language sampling (default: 42)")
    parser.add_argument("--question_type", choices=["auto", "open_ended", "mc", "math"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    overrides = {
        "dataset_path": args.dataset,
        "output_dir": args.output_dir,
        "baseline_type": args.baseline_type,
        "output_file": args.output_file,
        "translation_file": args.translation_file,
        "intermediate_file": args.intermediate_file,
        "num_samples": args.num_samples,
        "question_type": args.question_type,
    }
    for key, val in overrides.items():
        if val is not None:
            config_dict[key] = val

    config = load_config_from_dict(config_dict)

    translation_path = os.path.join(config.output_dir, config.translation_file)
    if os.path.exists(translation_path):
        print(f"Loading existing translations from {translation_path}")
        dataset = load_json_dataset(translation_path)
    else:
        raw_dataset = load_json_dataset(config.dataset_path)
        if args.num_samples_per_lang:
            raw_dataset = sample_per_language(raw_dataset, args.num_samples_per_lang, args.sample_seed)
            print(f"Sampled {len(raw_dataset)} items ({args.num_samples_per_lang} per language, seed={args.sample_seed})")
        dataset = translate_questions(config, raw_dataset)

    if args.num_samples_per_lang and os.path.exists(translation_path):
        dataset = sample_per_language(dataset, args.num_samples_per_lang, args.sample_seed)
        print(f"Sampled {len(dataset)} items from existing translations ({args.num_samples_per_lang} per language)")

    evaluate_pipeline(config, dataset)


if __name__ == "__main__":
    main()
