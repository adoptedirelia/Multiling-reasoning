"""
MT1-MT2 Pipeline Evaluation Main Code
"""
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional


from src.eval.utils import *


_OPTION_FIELDS = ["option_a", "option_b", "option_c", "option_d", "option_e"]
_ENGLISH_OPTION_FIELDS = ["english_option_a", "english_option_b", "english_option_c", "english_option_d", "english_option_e"]


def _extract_options(dataset: List[Dict]) -> Optional[List[Dict[str, str]]]:
    """Extract MC options from dataset. Returns None if not MC.
    Dynamically picks all option_a..option_e fields present in the data."""
    if not dataset or "option_a" not in dataset[0]:
        return None
    return [
        {field: s.get(field, "") for field in _OPTION_FIELDS}
        for s in dataset
    ]


def _extract_english_options(dataset: List[Dict]) -> Optional[List[Dict[str, str]]]:
    """Extract translated English MC options stored as english_option_a/b/c/d/e.
    Returns None if not present (e.g. non-MC data or old translation files)."""
    if not dataset or "english_option_a" not in dataset[0]:
        return None
    return [
        {field: s.get(field, "") for field in _OPTION_FIELDS}
        for s in [
            {f: item.get(ef, "") for f, ef in zip(_OPTION_FIELDS, _ENGLISH_OPTION_FIELDS)}
            for item in dataset
        ]
    ]


def _ckpt_path(path: str) -> str:
    """Derive a per-batch checkpoint path by inserting '.ckpt' before the extension."""
    base, ext = os.path.splitext(path)
    return f"{base}.ckpt{ext}"


def translate_questions(config: EvalConfig) -> List[str]:
    """
    Translate questions to English, with per-batch checkpointing so an interrupted
    run can resume without re-translating already-processed samples.

    Args:
        config: Evaluation configuration
    """
    print(f"Loading dataset from {config.dataset_path}...")
    dataset = load_json_dataset(config.dataset_path)

    question_type = resolve_question_type(config, dataset)
    is_mc = question_type == "mc"
    all_options = _extract_options(dataset) if is_mc else None

    all_questions = [sample.get('question', '') for sample in dataset]
    all_languages = [sample.get('language', '') for sample in dataset]
    all_ground_truths = [sample.get('answer', '') for sample in dataset]

    translation_path = os.path.join(config.output_dir, config.translation_file)
    ckpt_path = _ckpt_path(translation_path)

    # Load any previously translated samples (flat list with sample_idx).
    ckpt_data: List[Dict] = _load_checkpoint(ckpt_path)
    start_idx = len(ckpt_data)
    if start_idx > 0:
        print(f"Resuming MT1 translation from checkpoint: {start_idx}/{len(dataset)} samples already done.")

    if start_idx < len(dataset):
        mt1_engine = create_engine(config.mt1_config)
        mt1_engine.load_model()

        for i in range(start_idx, len(all_questions), config.batch_size):
            batch_end = min(i + config.batch_size, len(all_questions))
            batch_questions = all_questions[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None

            batch_outputs = translate_questions_batch(
                mt1_engine,
                batch_questions,
                config.mt1_config,
                batch_size=len(batch_questions),
                options=batch_opts,
            )

            for j, output in enumerate(batch_outputs):
                idx = i + j
                entry: Dict = {
                    'sample_idx': idx,
                    'question': all_questions[idx],
                    'english_question': output.get('translation', ''),
                    'answer': all_ground_truths[idx],
                }
                if is_mc:
                    entry.update(all_options[idx])
                    for field in _OPTION_FIELDS:
                        entry[f"english_{field}"] = output.get(field, '')
                ckpt_data.append(entry)

            _save_checkpoint(ckpt_data, ckpt_path)
            print(f"  Saved MT1 checkpoint: {len(ckpt_data)}/{len(dataset)} samples")

        del mt1_engine

    print(f"Translation completed. {len(ckpt_data)} questions translated.")

    # Rebuild the per-language dict expected by the rest of the pipeline.
    translation_results: Dict = {}
    for entry in ckpt_data:
        lang = all_languages[entry['sample_idx']]
        if lang not in translation_results:
            translation_results[lang] = []
        # Store without internal bookkeeping fields.
        final_entry = {k: v for k, v in entry.items() if k != 'sample_idx'}
        translation_results[lang].append(final_entry)

    print(f"Saving translation results to {translation_path}...")
    save_results(translation_results, translation_path)
    return load_json_dataset(translation_path)
    

def _load_checkpoint(output_path: str) -> List[Dict]:
    """Load existing results from a checkpoint file for resuming."""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data
    return []


def _save_checkpoint(results: List[Dict], output_path: str):
    """Incrementally save results to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def evaluate_pipeline(config: EvalConfig, translation_results: Dict[str, Dict[str, str]]):
    """
    Run evaluation pipeline with different baseline types.
    Supports incremental saving and checkpoint-based resuming.
    """
    print(f"Loading dataset from {config.dataset_path}...")

    dataset = translation_results
    
    if config.num_samples:
        dataset = dataset[:config.num_samples]
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Baseline type: {config.baseline_type}")

    question_type = resolve_question_type(config, dataset)
    is_mc = question_type == "mc"
    all_options = _extract_options(dataset) if is_mc else None
    english_options = _extract_english_options(dataset) if is_mc else None

    baseline_type = config.baseline_type.lower()
    output_path = os.path.join(config.output_dir, config.output_file)

    all_questions = [sample.get('question', '') for sample in dataset]
    all_languages = [sample.get('language', '') for sample in dataset]
    all_ground_truths = [sample.get('answer', '') for sample in dataset]
    english_questions = [sample.get('english_question', '') for sample in dataset]

    results = _load_checkpoint(output_path)
    start_idx = len(results)
    if start_idx > 0:
        print(f"Resuming from checkpoint: {start_idx}/{len(dataset)} samples already done.")

    batch_size = config.batch_size
    
    if baseline_type == "end_to_end":
        print(f"Loading model: {config.mt1_config.model_name}...")
        engine = create_engine(config.mt1_config)
        engine.load_model()

        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            questions = all_questions[i:batch_end]
            languages = all_languages[i:batch_end]
            ground_truths = all_ground_truths[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            
            batch_results = run_end_to_end(engine, questions, languages, config.mt1_config, config, options=batch_opts)
            for j in range(len(questions)):
                result = {
                    'sample_idx': i + j,
                    'question': questions[j],
                    'language': languages[j],
                    'ground_truth': ground_truths[j],
                    'result': batch_results[j]
                }
                results.append(result)


            _save_checkpoint(results, output_path)
            print(f"  Saved checkpoint: {len(results)}/{len(dataset)} samples")
    
    elif baseline_type == "cascade":
        intermediate_path = os.path.join(config.output_dir, config.intermediate_file)
        all_llm_results: List[Dict] = _load_checkpoint(intermediate_path)
        start_reasoning_idx = len(all_llm_results)

        if start_reasoning_idx >= len(dataset):
            print(f"Step 1: Loaded all {len(all_llm_results)} reasoning results from checkpoint.")
        else:
            if start_reasoning_idx == 0:
                print("Step 1: Running reasoning on all questions...")
            else:
                print(f"Step 1: Resuming reasoning from checkpoint: "
                      f"{start_reasoning_idx}/{len(dataset)} samples already done.")

            llm_engine = create_engine(config.llm_config)
            llm_engine.load_model()
            for i in range(start_reasoning_idx, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_english_questions = english_questions[i:batch_end]
                batch_opts = english_options[i:batch_end] if english_options else None
                batch_llm_results = run_reasoning(
                    llm_engine,
                    batch_english_questions,
                    model_config=config.llm_config,
                    eval_config=config,
                    options=batch_opts,
                )
                all_llm_results.extend(batch_llm_results)
                _save_checkpoint(all_llm_results, intermediate_path)
                print(f"  Saved intermediate checkpoint: {len(all_llm_results)}/{len(dataset)} samples")

            print(f"Reasoning completed. Processed {len(all_llm_results)} samples.")
            del llm_engine

        print("Step 2: Running MT2 on all reasoning results...")
        if config.lora_path:
            config.mt2_config.lora_path = config.lora_path
        mt2_engine = create_engine(config.mt2_config)
        mt2_engine.load_model()

        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_questions = all_questions[i:batch_end]
            batch_english_questions = english_questions[i:batch_end]
            batch_languages = all_languages[i:batch_end]
            batch_llm_results = all_llm_results[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            
            batch_mt2_results = run_mt2_base(
                engine=mt2_engine,
                questions=batch_questions,
                english_questions=batch_english_questions,
                english_answers=[r.get("answer", "") for r in batch_llm_results],
                languages=batch_languages,
                model_config=config.mt2_config,
                eval_config=config,
                options=batch_opts
            )
            
            for j in range(len(batch_questions)):
                result = {
                    'sample_idx': i + j,
                    'question': batch_questions[j],
                    'language': batch_languages[j],
                    'ground_truth': all_ground_truths[i + j],
                    'english_question': batch_english_questions[j],
                    'llm_result': batch_llm_results[j],
                    'mt2_result': batch_mt2_results[j]
                }
                results.append(result)

            _save_checkpoint(results, output_path)
            print(f"  Saved checkpoint: {len(results)}/{len(dataset)} samples")
    
    elif baseline_type == "prompting":
        intermediate_path = os.path.join(config.output_dir, config.intermediate_file)
        all_llm_results: List[Dict] = _load_checkpoint(intermediate_path)
        start_reasoning_idx = len(all_llm_results)

        if start_reasoning_idx >= len(dataset):
            print(f"Step 1: Loaded all {len(all_llm_results)} reasoning results from checkpoint.")
        else:
            if start_reasoning_idx == 0:
                print("Step 1: Running reasoning on all questions...")
            else:
                print(f"Step 1: Resuming reasoning from checkpoint: "
                      f"{start_reasoning_idx}/{len(dataset)} samples already done.")

            llm_engine = create_engine(config.llm_config)
            llm_engine.load_model()
            for i in range(start_reasoning_idx, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_english_questions = english_questions[i:batch_end]
                batch_opts = english_options[i:batch_end] if english_options else None
                batch_llm_results = run_reasoning(
                    llm_engine,
                    batch_english_questions,
                    model_config=config.llm_config,
                    eval_config=config,
                    options=batch_opts,
                )
                all_llm_results.extend(batch_llm_results)
                _save_checkpoint(all_llm_results, intermediate_path)
                print(f"  Saved intermediate checkpoint: {len(all_llm_results)}/{len(dataset)} samples")

            print(f"Reasoning completed. Processed {len(all_llm_results)} samples.")
            del llm_engine

        print("Step 2: Running MT2 on all reasoning results...")
        if config.lora_path:
            config.mt2_config.lora_path = config.lora_path
        mt2_engine = create_engine(config.mt2_config)
        mt2_engine.load_model()

        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_questions = all_questions[i:batch_end]
            batch_english_questions = english_questions[i:batch_end]
            batch_languages = all_languages[i:batch_end]
            batch_llm_results = all_llm_results[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None
            
            batch_mt2_results = run_mt2(
                engine=mt2_engine,
                questions=batch_questions,
                english_questions=batch_english_questions,
                english_thinking_processes=[r.get("reasoning", "") for r in batch_llm_results],
                english_answers=[r.get("answer", "") for r in batch_llm_results],
                languages=batch_languages,
                model_config=config.mt2_config,
                eval_config=config,
                options=batch_opts
            )

            for j in range(len(batch_questions)):
                result = {
                    'sample_idx': i + j,
                    'question': batch_questions[j],
                    'language': batch_languages[j],
                    'ground_truth': all_ground_truths[i + j],
                    'english_question': batch_english_questions[j],
                    'llm_result': batch_llm_results[j],
                    'mt2_result': batch_mt2_results[j]
                }
                results.append(result)

            _save_checkpoint(results, output_path)
            print(f"  Saved checkpoint: {len(results)}/{len(dataset)} samples")

    elif baseline_type.startswith("ablation_"):
        # Ablation studies: same two-step flow as prompting, but MT2 receives only a subset
        # of context. The sub-type (after "ablation_") controls what MT2 sees:
        #   answer_only      — English answer only
        #   answer_orig_q    — original question + English answer
        #   answer_eng_q     — English question + English answer
        #   answer_reasoning — English reasoning trace + English answer
        ablation_sub = baseline_type[len("ablation_"):]
        valid_ablations = {"answer_only", "answer_orig_q", "answer_eng_q", "answer_reasoning"}
        if ablation_sub not in valid_ablations:
            raise ValueError(
                f"Unknown ablation sub-type '{ablation_sub}'. "
                f"Valid options: {sorted(valid_ablations)}"
            )

        intermediate_path = os.path.join(config.output_dir, config.intermediate_file)
        all_llm_results: List[Dict] = _load_checkpoint(intermediate_path)
        start_reasoning_idx = len(all_llm_results)

        if start_reasoning_idx >= len(dataset):
            print(f"Step 1: Loaded all {len(all_llm_results)} reasoning results from checkpoint.")
        else:
            if start_reasoning_idx == 0:
                print("Step 1: Running reasoning on all questions...")
            else:
                print(f"Step 1: Resuming reasoning from checkpoint: "
                      f"{start_reasoning_idx}/{len(dataset)} samples already done.")

            llm_engine = create_engine(config.llm_config)
            llm_engine.load_model()
            for i in range(start_reasoning_idx, len(dataset), batch_size):
                batch_end = min(i + batch_size, len(dataset))
                batch_english_questions = english_questions[i:batch_end]
                batch_opts = english_options[i:batch_end] if english_options else None
                batch_llm_results = run_reasoning(
                    llm_engine,
                    batch_english_questions,
                    model_config=config.llm_config,
                    eval_config=config,
                    options=batch_opts,
                )
                all_llm_results.extend(batch_llm_results)
                _save_checkpoint(all_llm_results, intermediate_path)
                print(f"  Saved intermediate checkpoint: {len(all_llm_results)}/{len(dataset)} samples")

            print(f"Reasoning completed. Processed {len(all_llm_results)} samples.")
            del llm_engine

        print(f"Step 2: Running MT2 ablation '{ablation_sub}' on all reasoning results...")
        if config.lora_path:
            config.mt2_config.lora_path = config.lora_path
        mt2_engine = create_engine(config.mt2_config)
        mt2_engine.load_model()

        for i in range(start_idx, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_questions = all_questions[i:batch_end]
            batch_english_questions = english_questions[i:batch_end]
            batch_languages = all_languages[i:batch_end]
            batch_llm_results = all_llm_results[i:batch_end]
            batch_opts = all_options[i:batch_end] if all_options else None

            batch_mt2_results = run_mt2_ablation(
                engine=mt2_engine,
                ablation_type=ablation_sub,
                questions=batch_questions,
                english_questions=batch_english_questions,
                english_thinking_processes=[r.get("reasoning", "") for r in batch_llm_results],
                english_answers=[r.get("answer", "") for r in batch_llm_results],
                languages=batch_languages,
                model_config=config.mt2_config,
                eval_config=config,
                options=batch_opts,
            )

            for j in range(len(batch_questions)):
                result = {
                    'sample_idx': i + j,
                    'question': batch_questions[j],
                    'language': batch_languages[j],
                    'ground_truth': all_ground_truths[i + j],
                    'english_question': batch_english_questions[j],
                    'llm_result': batch_llm_results[j],
                    'mt2_result': batch_mt2_results[j],
                }
                results.append(result)

            _save_checkpoint(results, output_path)
            print(f"  Saved checkpoint: {len(results)}/{len(dataset)} samples")

    print(f"\nEvaluation completed! Processed {len(results)} samples.")
    print(f"Results saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MT1-MT2 Pipeline Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path (JSON format)")
    parser.add_argument("--dataset", type=str, help="Dataset path (overrides config file)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config file)")
    parser.add_argument("--mt1_model", type=str, help="MT1 model name (overrides config file)")
    parser.add_argument("--mt2_model", type=str, help="MT2 model name (overrides config file)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate (overrides config file)")
    parser.add_argument("--baseline_type", type=str,
                        help="Baseline type: end_to_end, cascade, prompting, "
                             "ablation_answer_only, ablation_answer_orig_q, "
                             "ablation_answer_eng_q, ablation_answer_reasoning")
    parser.add_argument("--output_file", type=str, help="Output file name (overrides config file)")
    parser.add_argument("--translation_file", type=str, help="Translation file name (overrides config file)")
    parser.add_argument("--save_intermediate", type=bool, help="Save intermediate results (overrides config file)")
    parser.add_argument("--lora_path", type=str, help="Lora path (overrides config file)")
    parser.add_argument("--intermediate_file", type=str, help="Intermediate file name (overrides config file)")
    parser.add_argument("--question_type", type=str, choices=["auto", "open_ended", "mc", "math"],
                        help="Question type: auto (detect from data), open_ended, mc, or math (overrides config file)")

    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Override configuration
    if args.dataset:
        config_dict['dataset_path'] = args.dataset
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
    if args.mt1_model:
        config_dict['mt1_config']['model_name'] = args.mt1_model
    if args.mt2_model:
        if 'mt2_config' not in config_dict:
            config_dict['mt2_config'] = config_dict['mt1_config'].copy()
        config_dict['mt2_config']['model_name'] = args.mt2_model
    if args.num_samples:
        config_dict['num_samples'] = args.num_samples
    if args.baseline_type:
        config_dict['baseline_type'] = args.baseline_type
    if args.output_file:
        config_dict['output_file'] = args.output_file
    if args.translation_file:
        config_dict['translation_file'] = args.translation_file
    if args.save_intermediate:
        config_dict['save_intermediate'] = args.save_intermediate
    if args.lora_path:
        config_dict['lora_path'] = args.lora_path
    if args.intermediate_file:
        config_dict['intermediate_file'] = args.intermediate_file
    if args.question_type:
        config_dict['question_type'] = args.question_type
    config = load_config_from_dict(config_dict)

    # Run evaluation

    if os.path.exists(os.path.join(config.output_dir, config.translation_file)):
        print(f"Translation results already exist at {os.path.join(config.output_dir, config.translation_file)}")
        translation_results = load_json_dataset(os.path.join(config.output_dir, config.translation_file))
    else:
        print(f"Translation results do not exist at {os.path.join(config.output_dir, config.translation_file)}")
        translation_results = translate_questions(config)
        save_results(translation_results, os.path.join(config.output_dir, config.translation_file))

    evaluate_pipeline(config, translation_results)


if __name__ == "__main__":
    main()
