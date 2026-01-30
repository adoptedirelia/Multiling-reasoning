from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainConfig:
    """Training configuration"""
    # Model configuration
    model_name: str  # Base model name or path
    output_dir: str  # Output directory
    
    # Data configuration
    train_data_path: str  # Training data path
    val_data_path: Optional[str] = None  # Validation data path
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Sequence length
    max_length: int = 4096
    
    # Optimizer configuration
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # "linear", "cosine", "constant"
    
    # Save and evaluation
    save_steps: int = 500
    eval_steps: Optional[int] = 500
    save_total_limit: int = 3  # Maximum number of checkpoints to save
    
    # Other configuration
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    logging_steps: int = 10
    
    # Device configuration
    device_map: str = "auto"
    torch_dtype: str = "auto"
    
    # DeepSpeed configuration
    deepspeed_config_path: Optional[str] = 'configs/deepspeed_config_zero2_no_offload.json'  # Path to DeepSpeed config file


def load_train_config_from_dict(config_dict: Dict[str, Any]) -> TrainConfig:
    """Load training configuration from dictionary"""
    return TrainConfig(**config_dict)
