"""
MT2 Training Dataset
"""
import json
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from src.prompt import MT2_PROMPT


class MT2Dataset(Dataset):
    """MT2 Training Dataset
    
    Input: Target language question + English CoT
    Output: Target language CoT + Target language answer
    """
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, 
                 max_length: int = 4096, is_training: bool = True):
        """
        Initialize dataset
        
        Args:
            data_path: Data file path (JSON format)
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            is_training: Whether in training mode
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # Load data
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If data is in dictionary format, convert to list
        if isinstance(data, dict):
            result = []
            for key, value in data.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            item['language'] = key
                        result.append(item)
            return result
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported data format in {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a single sample"""
        sample = self.data[idx]
        
        # Get input
        question = sample.get('question', '')
        language = sample.get('language', '')
        english_question = sample.get('english_question', '')
        english_answer = sample.get('english_answer', '')
        mt1_result = sample.get('llm_result', '')
        english_reasoning = mt1_result.get('reasoning', '') or mt1_result.get('English_Thinking_Process', '')
        english_answer = mt1_result.get('answer', '')

        # Build input prompt
        input_text = MT2_PROMPT.format(
            question=question,
            language=language,
            English_Question=english_question,
            English_Thinking_Process=english_reasoning,
            English_Answer=english_answer
        )

        
        # Build output text
        output_text = f"<answer>{sample.get('ground_truth', '')}</answer>"
        
        # Build full text (for training)
        full_text = input_text + "\n" + output_text
        
        dict = {
            'input_text': input_text,
            'output_text': output_text,
            'full_text': full_text,
            'question': question,
            'language': language
        }
        
        return dict
    
    def collate_fn(self, batch: List[Dict[str, str]]) -> Dict:
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    dataset = MT2Dataset(data_path="results/train_example_GPT.json", tokenizer=tokenizer, max_length=2048, is_training=True)

    res = []
    ratio = 0.9
    type_list ={}
    for i in dataset:
        language = i['language']
        if language not in type_list:
            type_list[language] = []
        type_list[language].append(i)
        # res.append({'input':i['input_text'], 'answer':i['output_text']})
    train_data_out = []
    val_data_out = []
    for language, data in type_list.items():
        train_data = data[:int(len(data) * ratio)]
        val_data = data[int(len(data) * ratio):]
        train_data_out.extend([{'input': item['input_text'], 'answer': item['output_text']} for item in train_data])
        val_data_out.extend([{'input': item['input_text'], 'answer': item['output_text']} for item in val_data])
    import os
    if os.path.exists('results/train.jsonl'):
        os.remove('results/train.jsonl')
    if os.path.exists('results/val.jsonl'):
        os.remove('results/val.jsonl')
    with open('results/train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data_out:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open('results/val.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data_out:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(val_data_out[0])