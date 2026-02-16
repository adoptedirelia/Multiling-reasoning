
"""
MT1-MT2 Pipeline Evaluation Main Code
"""
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional


from src.eval.utils import *

def translate_questions(config: EvalConfig) -> List[str]:
    """
    Translate questions to English
    
    Args:
        config: Evaluation configuration
    """
    print(f"Loading dataset from {config.dataset_path}...")
    dataset = load_json_dataset(config.dataset_path)
    
    
    # Extract all questions and languages
    all_questions = [sample.get('question', '') for sample in dataset]
    all_languages = [sample.get('language', '') for sample in dataset]
    all_ground_truths = [sample.get('answer', '') for sample in dataset]

    mt1_engine = create_engine(config.mt1_config)
    mt1_engine.load_model()
    
    # Translate all questions in batches
    english_questions = translate_questions_batch(
        mt1_engine, 
        all_questions, 
        config.mt1_config,
        batch_size=config.batch_size
    )
    print(f"Translation completed. Translated {len(english_questions)} questions.")
    
    # Save translation results if needed

    translation_results = {}
    for i in range(len(dataset)):
        
        language = all_languages[i]
        if language not in translation_results.keys():
            translation_results[language] = []
        translation_results[language].append({
            'question': all_questions[i],
            'english_question': english_questions[i],
            'answer': all_ground_truths[i]
        })
    translation_path = os.path.join(config.output_dir, config.translation_file)
    print(f"Saving translation results to {translation_path}...")
    save_results(translation_results, translation_path)
    res = load_json_dataset(translation_path)
    return res
    

def evaluate_pipeline(config: EvalConfig, translation_results: Dict[str, Dict[str, str]]):
    """
    Run evaluation pipeline with different baseline types
    
    Args:
        config: Evaluation configuration
    """
    # Load dataset
    print(f"Loading dataset from {config.dataset_path}...")
    if translation_results is None:
        dataset = load_json_dataset(config.dataset_path)
    else:
        dataset = translation_results
    
    if config.num_samples:
        dataset = dataset[:config.num_samples]
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Baseline type: {config.baseline_type}")
    
    baseline_type = config.baseline_type.lower()

    # Extract all questions and languages
    all_questions = [sample.get('question', '') for sample in dataset]
    all_languages = [sample.get('language', '') for sample in dataset]
    all_ground_truths = [sample.get('answer', '') for sample in dataset]
    english_questions = [sample.get('english_question', '') for sample in dataset]
    errors = [sample.get('error', {}) for sample in dataset]
    # Step 1: Translate all questions to English (MT1 step) - only for cascade and prompting baselines
    mt1_engine = create_engine(config.mt1_config)
    mt1_engine.load_model()
    # Initialize engines based on baseline type
    if baseline_type == "end_to_end":
        # End-to-end: only need one engine
        print(f"Loading model: {config.mt1_config.model_name}...")
        
        engine = mt1_engine
    elif baseline_type in ["cascade", "prompting"]:
        # LLM engine (for reasoning)
        if config.llm_config and config.llm_config.model_name != config.mt1_config.model_name:
            print(f"Loading LLM model: {config.llm_config.model_name}...")
            llm_engine = create_engine(config.llm_config)
            llm_engine.load_model()
        else:
            print("Using MT1 model as LLM model")
            llm_engine = mt1_engine
        
        # MT2 engine
        if config.mt2_config and config.mt2_config.model_name != config.mt1_config.model_name:
            print(f"Loading MT2 model: {config.mt2_config.model_name}...")
            mt2_engine = create_engine(config.mt2_config)
            mt2_engine.load_model()
        else:
            print("Using same model for MT1 and MT2")
            mt2_engine = mt1_engine
    else:
        raise ValueError(f"Unsupported baseline type: {config.baseline_type}")
    
    # Run evaluation
    results = []
    batch_size = config.batch_size
    
    for i in range(0, len(dataset), batch_size):
        batch_samples = dataset[i:i+batch_size]
        questions = all_questions[i:i+batch_size]
        languages = all_languages[i:i+batch_size]
        ground_truths = all_ground_truths[i:i+batch_size]
        
        if baseline_type == "end_to_end":
            # End-to-end baseline
            batch_results = run_end_to_end(engine, questions, languages, config.mt1_config, config)
            for j in range(len(batch_samples)):
                result = {
                    'sample_idx': i + j,
                    'question': questions[j],
                    'language': languages[j],
                    'ground_truth': ground_truths[j],
                    'result': batch_results[j],
                    'error': errors[j]
                }
                results.append(result)
        
        elif baseline_type == "cascade":
            # Cascade baseline: English questions -> LLM -> English output -> MT2 -> Output
            batch_english_questions = english_questions[i:i+batch_size]
            batch_results = run_cascade_baseline(
                llm_engine, mt2_engine,
                questions, batch_english_questions, languages,
                config.llm_config, config.mt2_config, config
            )
            for j in range(len(batch_samples)):
                result = {
                    'sample_idx': i + j,
                    'question': questions[j],
                    'language': languages[j],
                    'ground_truth': ground_truths[j],
                    'english_question': batch_results[j].get('english_question', ''),
                    'llm_result': batch_results[j].get('llm_result', {}),
                    'mt2_result': batch_results[j].get('mt2_result', {}),
                    'error': errors[j]
                }
                results.append(result)
        
        elif baseline_type == "prompting":
            # Prompting baseline: English questions -> LLM -> (input, eng reasoning, eng output) -> MT2 -> Output
            batch_english_questions = english_questions[i:i+batch_size]
            batch_results = run_prompting_baseline_with_error(
                llm_engine, mt2_engine,
                questions, batch_english_questions, languages,
                config.llm_config, config.mt2_config, config,
                mt1_result=batch_samples
            )
            for j in range(len(batch_samples)):
                result = {
                    'sample_idx': i + j,
                    'question': questions[j],
                    'language': languages[j],
                    'ground_truth': ground_truths[j],
                    'english_question': batch_results[j].get('english_question', ''),
                    'llm_result': batch_results[j].get('llm_result', {}),
                    'mt2_result': batch_results[j].get('mt2_result', {}),
                    'error': errors[j]
                }
                results.append(result)
        

    # Save results
    output_path = os.path.join(config.output_dir, config.output_file)
    print(f"Saving results to {output_path}...")
    save_results(results, output_path)
    
    print(f"Evaluation completed! Processed {len(results)} samples.")


def main(error_data):
    """Main function"""
    parser = argparse.ArgumentParser(description="MT1-MT2 Pipeline Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path (JSON format)")
    parser.add_argument("--dataset", type=str, help="Dataset path (overrides config file)")
    parser.add_argument("--output_dir", type=str, help="Output directory (overrides config file)")
    parser.add_argument("--mt1_model", type=str, help="MT1 model name (overrides config file)")
    parser.add_argument("--mt2_model", type=str, help="MT2 model name (overrides config file)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to evaluate (overrides config file)")
    parser.add_argument("--baseline_type", type=str, help="Baseline type (end_to_end, cascade, prompting)")
    parser.add_argument("--output_file", type=str, help="Output file name (overrides config file)")
    
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

    config = load_config_from_dict(config_dict)


    evaluate_pipeline(config, error_data)


def find_examples(id, dataset, language) -> List[Dict[str, str]]:
    # unfold_data = [i for item in dataset.values() for i in item]
    for data in dataset[language]:
        if data['example_id'] == id:
            return data

if __name__ == "__main__":

    language_dict = {
        'am': 'Amharic',
        'ar': 'Arabic',
        'ja': 'Japanese',
        'zh_cn': 'Chinese',
        'mr': 'Marathi',
        'vi': 'Vietnamese',
        'te': 'Telugu',
    }

    dataset = '/home/dzhang98/code/Multiling-reasoning/dataset/PIQA.json'
    with open(dataset, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    error_data_path = '/home/dzhang98/code/Multiling-reasoning/data_transfer/global_piqa_error_sim.jsonl'
    error_data = []
    with open(error_data_path, 'r', encoding='utf-8') as f:
        for line in f:

            error = json.loads(line)
            example_id = error['example_id']
            error_group = error['error_group']
            data = find_examples(example_id, dataset, language_dict[error['lang']])
            if error_group == 'input_err':
                continue
            else:
                llm_result = {
                    'question': error['x_en'],
                    'reasoning': error['r_en_err'],
                    'answer': error['y_en_err'],
                }
                data['llm_result'] = llm_result
                data['error'] = error
                error_data.append(data.copy())

    main(error_data)
