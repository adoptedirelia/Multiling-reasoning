#!/usr/bin/env python
"""
Training script that can be run with DeepSpeed or torchrun
Usage:
    # Method 1: Using DeepSpeed launcher (recommended)
    deepspeed --num_gpus=<num_gpus> train.py --config <config_file>
    
    # Method 2: Using torchrun (also works with DeepSpeed)
    torchrun --nproc_per_node=<num_gpus> train.py --config <config_file>
    
    # Method 3: Single GPU training (without DeepSpeed)
    python train.py --config <config_file>
"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train.config import TrainConfig
from src.train.trainer import MT2Trainer


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MT2 Model Training")
    parser.add_argument("--config", type=str, required=True, help="Training configuration file path (JSON format)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = TrainConfig(**config_dict)
    
    # Create trainer and start training
    trainer = MT2Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
