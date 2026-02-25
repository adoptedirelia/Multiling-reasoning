import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from config import TrainConfig
import json
import warnings
warnings.filterwarnings("ignore")


from transformers import TrainerCallback
import torch

import torch

class SimpleCompletionOnlyCollator:
    def __init__(self, tokenizer, response_template):
        self.tokenizer = tokenizer
        self.response_template = response_template

    def find_position(self, input_ids, response_ids):
        input_ids = input_ids.tolist()
        for i in range(len(input_ids)):
            if input_ids[i:i+len(response_ids)] == response_ids:
                return i
        return None


    def __call__(self, examples):


        messages = [example["message"] for example in examples]
        inputs = self.tokenizer(messages, add_special_tokens=True,return_tensors="pt",padding=True)



        labels = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        
        for i in range(len(inputs["input_ids"])):
            
            
            inputs_ids = inputs["input_ids"][i]
            response_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
            start_idx = self.find_position(inputs_ids, response_ids)


            
            if start_idx == -1:
                labels[i][:] = -100
                print(f"Warning: Could not find assistant start token in sample {i}")
            else:
                labels[i][:start_idx+2] = -100

            

            if attention_mask is not None:

                labels[i][attention_mask[i] == 0] = -100

        inputs["labels"] = labels


        return inputs

class MT2Trainer:
    def __init__(self, config: TrainConfig):

        self.config = config
        self.model_name = config.model_name
        self.ANSWER_PREFIX = "Output:\n"
        # self.ANSWER_PREFIX = "[Your output goes here]"
        self.ANSWER_BEGIN = "<answer>"
        self.ANSWER_END = "</answer>"


    def setup_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device_map if self.config.deepspeed_config_path is None else None,
        )

        return model, tokenizer

    def build_text(self, example):
        prompt = example["input"].rstrip()

        ans = example["answer"].strip()
        if not ans.startswith(self.ANSWER_BEGIN):
            ans = f"{self.ANSWER_BEGIN}{ans}{self.ANSWER_END}"
        # return {"prompt": prompt, "completion": ans}
        return {"message": prompt + "\n" + ans}

    def prepare_datasets(self):
        train_dataset = load_dataset("json", data_files=self.config.train_data_path, split="train")
        val_dataset = load_dataset("json", data_files=self.config.val_data_path, split="train")
        train_dataset = train_dataset.map(self.build_text, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(self.build_text, remove_columns=val_dataset.column_names)

        return train_dataset, val_dataset
        
    def train(self):
        train_dataset, val_dataset = self.prepare_datasets()

        model, tokenizer = self.setup_model()

        # data_collator = DataCollatorForCompletionOnlyLM(
        #     response_template=self.ANSWER_PREFIX,
        #     tokenizer=tokenizer,
        # )



        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        args = SFTConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            report_to="none",
            deepspeed=self.config.deepspeed_config_path,

        )
        args.dataset_kwargs = {"skip_prepare_dataset": True}
        args.remove_unused_columns = False
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=SimpleCompletionOnlyCollator(tokenizer, self.ANSWER_PREFIX), 
            peft_config=lora_config if self.config.use_lora else None,
            args=args,
        )
        # sample_prompt = train_dataset[0]["text"].split("<answer>")[0]

        # trainer.add_callback(
        #     PrintOutputCallback(tokenizer, sample_prompt, interval=2)
        # )
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving final model to {self.config.output_dir}...")
        trainer.save_model()
        trainer.tokenizer.save_pretrained(self.config.output_dir)
        
        print("Training completed!")

def main():
    """Main function"""

    
    config = TrainConfig()
    
    # Create trainer and start training
    trainer = MT2Trainer(config)
    trainer.train()



if __name__ == "__main__":
    main()