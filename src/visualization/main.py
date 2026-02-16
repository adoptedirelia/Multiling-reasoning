"""
Multi-Model Attention Visualization Tool

This module provides structured classes and methods to:
1. Load various language models and tokenizers
2. Generate text and extract attention weights
3. Visualize attention weights

Supported models include:
- Llama models (meta-llama/Llama-3.1-8B-Instruct, etc.)
- Mistral models (mistralai/Mistral-7B-Instruct-v0.2, etc.)
- Qwen models (Qwen/Qwen2-7B-Instruct, etc.)
- And other AutoModelForCausalLM compatible models
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
import os
import argparse
try:
    from prompt import *
except ImportError:
    from ..prompt import *

from .utils import MODEL_CONFIGS, combine_attention_maps, list_available_models, ModelAttentionAnalyzer



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Model Attention Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python llama_test.py --list
  
  # Use predefined model configuration
  python llama_test.py --model-config mistral-7b --prompt "What is AI?"
  
  # Use custom model
  python llama_test.py --model-name "Qwen/Qwen2.5-7B-Instruct" --prompt "Hello"
  
  # Full example with all options
  python llama_test.py --model-config llama-3.1-8b \\
      --prompt "What is the capital of France?" \\
      --output-dir ./my_plots \\
      --max-tokens 256
  
  # Multi-part attention analysis (for MT2_PROMPT style prompts)
  python llama_test.py --multipart \\
      --prompt "$(cat formatted_prompt.txt)" \\
      --question "法国的首都是哪里？" \\
      --english-question "What is the capital of France?" \\
      --english-thinking "France is a country in Europe. The capital is Paris." \\
      --english-answer "Paris" \\
      --output-dir ./multipart_plots
        """
    )
    
    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-config", "-m",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        help="Predefined model configuration to use"
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="Custom model name or path (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    
    # Input/Output
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="What is the capital of China?",
        help="Input prompt text (default: 'What is the capital of China?')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./attention_plots",
        help="Output directory for attention plots (default: './attention_plots')"
    )
    
    # Multi-part analysis
    parser.add_argument(
        "--multipart",
        action="store_true",
        help="Enable multi-part attention analysis (requires --question, --english-question, etc.)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question text (for multi-part analysis)"
    )
    parser.add_argument(
        "--english-question",
        type=str,
        help="English question text (for multi-part analysis)"
    )
    parser.add_argument(
        "--english-thinking",
        type=str,
        help="English thinking process (for multi-part analysis)"
    )
    parser.add_argument(
        "--english-answer",
        type=str,
        help="English answer (for multi-part analysis)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cache during generation (default: True)"
    )
    parser.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable cache during generation"
    )
    
    # Model loading parameters
    parser.add_argument(
        "--attn-implementation",
        type=str,
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Attention implementation method"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: 'auto')"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch data type (default: 'auto')"
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=None,
        help="Force use of chat template"
    )
    parser.add_argument(
        "--no-chat-template",
        dest="use_chat_template",
        action="store_false",
        help="Disable chat template"
    )
    
    # Utility
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available predefined model configurations"
    )
    
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    """
    Main function - Run attention visualization
    
    Args:
        args: Parsed command line arguments (if None, will parse from sys.argv)
    """
    if args is None:
        args = parse_args()
    
    # Handle list command
    if args.list:
        list_available_models()
        return
    
    # Determine model configuration
    model_kwargs = {}
    
    if args.model_config:
        # Use predefined configuration
        print(f"=" * 60)
        print(f"Using predefined model configuration: {args.model_config}")
        print(f"=" * 60)
        model_kwargs["model_config"] = args.model_config
    elif args.model_name:
        # Use custom model
        print(f"=" * 60)
        print(f"Using custom model: {args.model_name}")
        print(f"=" * 60)
        model_kwargs["model_name"] = args.model_name
    else:
        # Use default
        print(f"=" * 60)
        print(f"Using default model (Llama-3.1-8B)")
        print(f"=" * 60)
    
    # Add optional parameters
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.torch_dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        model_kwargs["torch_dtype"] = dtype_map[args.torch_dtype]
    if args.use_chat_template is not None:
        model_kwargs["use_chat_template"] = args.use_chat_template
    
    # Initialize analyzer
    analyzer = ModelAttentionAnalyzer(**model_kwargs)
    
    # Check if multi-part analysis is requested
    if args.multipart:
        # Build parts dictionary
        parts = {}
        if args.question:
            parts["question"] = args.question
        if args.english_question:
            parts["English_Question"] = args.english_question
        if args.english_thinking:
            parts["English_Thinking_Process"] = args.english_thinking
        if args.english_answer:
            parts["English_Answer"] = args.english_answer
        
        if not parts:
            print("Error: --multipart requires at least one part (--question, --english-question, etc.)")
            return
        
        # Run multi-part visualization
        analyzer.visualize_multipart_attentions(
            full_prompt=args.prompt,
            parts=parts,
            output_dir=args.output_dir,
            max_new_tokens=args.max_tokens,
            use_cache=args.use_cache
        )
    else:
        # Run standard visualization
        analyzer.visualize_attentions(
            prompt=args.prompt,
            output_dir=args.output_dir,
            max_new_tokens=args.max_tokens,
            use_cache=args.use_cache
        )


def example_multipart_analysis():
    """
    Example function showing how to use multi-part attention analysis with MT2_PROMPT
    
    This demonstrates analyzing attention from the answer to:
    - question
    - English_Question
    - English_Thinking_Process
    - English_Answer
    """
    
    # Example data
    language = "Chinese"
    question = "法国的首都是哪里？"
    english_question = "What is the capital of France?"
    english_thinking = "France is a country in Europe. The capital city of France is Paris."
    english_answer = "Paris"
    
    # Format the prompt
    full_prompt = MT2_PROMPT.format(
        language=language,
        question=question,
        English_Question=english_question,
        English_Thinking_Process=english_thinking,
        English_Answer=english_answer
    )
    
    # Define parts for analysis
    parts = {
        "question": question,
        "English_Question": english_question,
        "English_Thinking_Process": english_thinking,
        "English_Answer": english_answer
    }
    
    # Initialize analyzer
    analyzer = ModelAttentionAnalyzer()
    
    # Run multi-part visualization
    analyzer.visualize_multipart_attentions(
        full_prompt=full_prompt,
        parts=parts,
        output_dir="~/code/Multiling-reasoning/src/visualization/attention_plots",
        max_new_tokens=512
    )



if __name__ == "__main__":
    args = parse_args()
    # main(args)
    example_multipart_analysis()