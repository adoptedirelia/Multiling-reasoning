"""
MT2 Training Dataset
"""
import json
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.prompt import MT2_PROMT


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
        mt1_result = sample.get('mt1_result', '')
        english_reasoning = mt1_result.get('reasoning', '') or mt1_result.get('English_Thinking_Process', '')
        
        # Build input prompt
        input_text = MT2_PROMT.format(
            question=question,
            language=language,
            English_Thinking_Process=english_reasoning
        ).replace("Output:", "").strip()  # Remove Output: as this is what we want to generate
        
        # Get output (target language CoT and answer)
        mt2_result = sample.get('mt2_result', '')
        target_reasoning = mt2_result.get('reasoning', '') or mt2_result.get('reasoning', '')
        target_answer = mt2_result.get('answer', '') or mt2_result.get('answer', '')
        
        # Build output text
        output_text = ""
        if target_reasoning:
            output_text += f"<think>{target_reasoning}</think>\n"
        if target_answer:
            output_text += f"<answer>{target_answer}</answer>"
        
        # Build full text (for training)
        full_text = input_text + "\n" + output_text
        
        return {
            'input_text': input_text,
            'output_text': output_text,
            'full_text': full_text,
            'question': question,
            'language': language
        }
    
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
