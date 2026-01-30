"""
MT1-MT2 Pipeline Evaluation Main Code
"""
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional

from src.eval.config import EvalConfig, ModelConfig, load_config_from_dict
from src.eval.engine import Qwen3Engine, BaseEngine
from src.eval.utils import extract_mt1_output, extract_mt2_output, load_json_dataset, save_results
from src.prompt import MT1_PROMT, MT2_PROMT


def create_engine(model_config: ModelConfig) -> BaseEngine:
    """
    Create model engine based on configuration
    
    Args:
        model_config: Model configuration
    
    Returns:
        Model engine instance
    """
    if model_config.model_type.lower() == "qwen3":
        return Qwen3Engine(
            model_name=model_config.model_name,
            device_map=model_config.device_map,
            torch_dtype=model_config.torch_dtype,
            attn_implementation=model_config.attn_implementation
        )
    else:
        raise ValueError(f"Unsupported model type: {model_config.model_type}")


def run_mt1(engine: BaseEngine, questions: str, languages: str = "", 
            model_config: Optional[ModelConfig] = None, eval_config: Optional[EvalConfig] = None) -> Dict[str, str]:
    """
    Run MT1 stage: Input target language question, output English CoT and answer
    
    Args:
        engine: Model engine
        question: Target language questions
        languages: Question languages (optional)
        model_config: Model configuration (optional)
    
    Returns:
        Dictionary containing English question, reasoning, answer
    """
    # Build MT1 prompt
    prompts = [MT1_PROMT.format(question=question) for question in questions]
    
    # Generation parameters
    gen_kwargs = {}
    if model_config:
        gen_kwargs['max_new_tokens'] = model_config.max_new_tokens
        gen_kwargs['temperature'] = model_config.temperature
        gen_kwargs['top_p'] = model_config.top_p
    
    # Generate
    outputs = engine.generate_batch(prompts=prompts, **gen_kwargs)
    
    # Extract structured information
    results = [extract_mt1_output(output) for output in outputs]
    return results


def run_mt2(engine: BaseEngine, questions: str, english_reasonings: str, 
            languages: str = "", model_config: Optional[ModelConfig] = None, eval_config: Optional[EvalConfig] = None) -> Dict[str, str]:
    """
    Run MT2 stage: Input target language question and English CoT, output target language CoT and answer
    
    Args:
        engine: Model engine
        questions: Target language questions
        english_reasoning: English reasoning process
        language: Target language
        model_config: Model configuration (optional)
    
    Returns:
        Dictionary containing target language reasoning and answer
    """
    # Build MT2 prompt
    prompts = []
    for i in range(len(questions)):
        prompt = MT2_PROMT.format(
            question=questions[i],
            language=languages[i],
            English_Thinking_Process=english_reasonings[i]
        )
        prompts.append(prompt)
    
    # Generation parameters
    gen_kwargs = {}
    if model_config:
        gen_kwargs['max_new_tokens'] = model_config.max_new_tokens
        gen_kwargs['temperature'] = model_config.temperature
        gen_kwargs['top_p'] = model_config.top_p
    
    # Generate
    outputs = engine.generate_batch(prompts=prompts, **gen_kwargs)
    
    # Extract structured information
    results = [extract_mt2_output(output) for output in outputs]
    return results


def evaluate_pipeline(config: EvalConfig):
    """
    Run complete MT1-MT2 evaluation pipeline
    
    Args:
        config: Evaluation configuration
    """
    # Load dataset
    print(f"Loading dataset from {config.dataset_path}...")
    dataset = load_json_dataset(config.dataset_path)
    
    if config.num_samples:
        dataset = dataset[:config.num_samples]
    
    print(f"Loaded {len(dataset)} samples")
    
    # Create MT1 engine
    print(f"Loading MT1 model: {config.mt1_config.model_name}...")
    mt1_engine = create_engine(config.mt1_config)
    mt1_engine.load_model()
    
    # Create MT2 engine (if different from MT1)
    if config.mt2_config and config.mt2_config.model_name != config.mt1_config.model_name:
        print(f"Loading MT2 model: {config.mt2_config.model_name}...")
        mt2_engine = create_engine(config.mt2_config)
        mt2_engine.load_model()
    else:
        print("Using same model for MT1 and MT2")
        mt2_engine = mt1_engine
    
    # Run evaluation
    results = []
    intermediate_results = [] if config.save_intermediate else None
    # batch evaluation
    batch_size = config.batch_size
    for i in range(0, len(dataset), batch_size):
        batch_samples = dataset[i:i+batch_size]
        questions = [sample.get('question', '') for sample in batch_samples]
        languages = [sample.get('language', '') for sample in batch_samples]
        ground_truths = [sample.get('answer', '') for sample in batch_samples]
        # mt1_results = mt1_engine.generate_batch(questions, language=languages, config.mt1_config)
        mt1_results = run_mt1(mt1_engine, questions, languages, config.mt1_config, config)

        if config.save_intermediate:
            for j in range(len(batch_samples)):
                intermediate_results.append({
                    'sample_idx': i + j,
                    'question': questions[j],
                    'language': languages[j],
                    'mt1_result': mt1_results[j]
                })
        mt2_results = run_mt2(mt2_engine, questions, mt1_results, languages, config.mt2_config, config)
        for j in range(len(batch_samples)):
            sample = batch_samples[j]
            idx = i + j
            question = sample.get('question', '')
            language = sample.get('language', '')
            ground_truth = sample.get('answer', '')
            mt1_result = mt1_results[j]
            mt2_result = mt2_results[j]
            result = {
                'sample_idx': idx,
                'question': question,
                'language': language,
                'ground_truth': ground_truth,
                'mt1_result': mt1_result,
                'mt2_result': mt2_result
            }
            results.append(result)



    # Save results
    output_path = os.path.join(config.output_dir, config.output_file)
    print(f"Saving results to {output_path}...")
    save_results(results, output_path)
    
    if config.save_intermediate and intermediate_results:
        intermediate_path = os.path.join(config.output_dir, "mt1_intermediate.json")
        print(f"Saving MT1 intermediate results to {intermediate_path}...")
        save_results(intermediate_results, intermediate_path)
    
    print(f"Evaluation completed! Processed {len(results)} samples.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MT1-MT2 Pipeline Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path (JSON format)")
    parser.add_argument("--dataset", type=str, help="Dataset path (overrides config file)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config file)")
    parser.add_argument("--mt1_model", type=str, help="MT1 model name (overrides config file)")
    parser.add_argument("--mt2_model", type=str, help="MT2 model name (overrides config file)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate (overrides config file)")
    
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
    
    config = load_config_from_dict(config_dict)
    
    # Run evaluation
    evaluate_pipeline(config)


if __name__ == "__main__":
    main()
