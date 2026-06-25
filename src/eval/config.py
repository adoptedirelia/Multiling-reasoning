from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str  # "qwen3", "openai", etc.
    model_name: str  # Model name or path (e.g. "gpt-4o" for OpenAI)
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"  # Only for qwen3
    max_new_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    lora_path: str = None
    # OpenAI-compatible API settings
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    timeout_s: int = 60


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # MT1 model configuration
    mt1_config: ModelConfig
    
    # MT2 model configuration (if different from MT1)
    mt2_config: Optional[ModelConfig] = None
    
    # LLM model configuration (for reasoning, used in cascade and prompting baselines)
    llm_config: Optional[ModelConfig] = None
    
    # Dataset configuration
    dataset_path: str = ""
    dataset_type: str = "json"  # "json" or other types
    
    # Output configuration
    output_dir: str = "./results"
    output_file: str = "eval_results.json"
    translation_file: str = "mt1_translations.json"
    intermediate_file: str = "intermediate_results.json"
    lora_path: str = None
    
    # Baseline type: "end_to_end", "cascade", "prompting", or "mt1_mt2" (default pipeline)
    baseline_type: str = "mt1_mt2"
    
    # Question type: "auto" (detect from data), "open_ended", or "mc" (multiple choice)
    question_type: str = "auto"
    
    # Other configuration
    batch_size: int = 1
    num_samples: Optional[int] = None  # If specified, only evaluate first N samples
    save_intermediate: bool = True  # Whether to save MT1 intermediate results


def load_config_from_dict(config_dict: Dict[str, Any]) -> EvalConfig:
    """Load configuration from dictionary"""
    mt1_dict = config_dict.get("mt1_config", {})
    mt1_config = ModelConfig(**mt1_dict)
    
    mt2_dict = config_dict.get("mt2_config")
    mt2_config = ModelConfig(**mt2_dict) if mt2_dict else None
    
    llm_dict = config_dict.get("llm_config")
    llm_config = ModelConfig(**llm_dict) if llm_dict else None
    
    eval_dict = {k: v for k, v in config_dict.items() if k not in ["mt1_config", "mt2_config", "llm_config"]}
    return EvalConfig(mt1_config=mt1_config, mt2_config=mt2_config, llm_config=llm_config, **eval_dict)
