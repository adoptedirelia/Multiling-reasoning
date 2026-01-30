"""
MT2 Trainer
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Optional

from src.train.config import TrainConfig
from src.train.dataset import MT2Dataset

from typing import List, Dict

class CollateFn:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict:
        """Batch processing function"""
        # Extract texts
        texts = [item['full_text'] for item in batch]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids, but need to mask input part)
        labels = encodings['input_ids'].clone()
        
        # Find position after "Output:" in each sample, only calculate loss for that part
        for i, item in enumerate(batch):
            input_text = item['input_text']
            # Tokenize input part
            input_encodings = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_length = input_encodings['input_ids'].shape[1]
            
            # Set labels for input part to -100 (ignore loss)
            labels[i, :input_length] = -100
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }



class MT2Trainer:
    """MT2 Model Trainer"""
    
    def __init__(self, config: TrainConfig):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_model(self):
        """Setup model and tokenizer"""
        print(f"Loading model and tokenizer from {self.config.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        dtype = dtype_map.get(self.config.torch_dtype, "auto")
        
        # When using DeepSpeed, device_map should be None or not set
        # DeepSpeed will handle device placement
        device_map = None if self.config.deepspeed_config_path else self.config.device_map
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype if dtype != "auto" else "auto",
            device_map=device_map
        )
        
        if self.config.deepspeed_config_path:
            print(f"DeepSpeed enabled with config: {self.config.deepspeed_config_path}")
        print("Model and tokenizer loaded successfully!")
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        print("Preparing datasets...")
        
        train_dataset = MT2Dataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            is_training=True
        )
        
        val_dataset = None
        if self.config.val_data_path:
            val_dataset = MT2Dataset(
                data_path=self.config.val_data_path,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                is_training=False
            )
        
        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"Val dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self):
        """Start training"""
        # Setup model
        self.setup_model()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling, not masked language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            lr_scheduler_type=self.config.lr_scheduler_type,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            logging_steps=self.config.logging_steps,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            remove_unused_columns=False,
            report_to="none",  # Don't use wandb etc.
            deepspeed=self.config.deepspeed_config_path,  # DeepSpeed config file path
        )
        
        # Create Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=CollateFn(self.tokenizer, self.config.max_length),
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("Starting training...")
        self.trainer.train()
        
        # Save final model
        print(f"Saving final model to {self.config.output_dir}...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("Training completed!")


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
