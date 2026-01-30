from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseEngine(ABC):
    """Base model engine interface"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize model engine
        
        Args:
            model_name: Model name or path
            **kwargs: Other model parameters
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self):
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            **kwargs: Other generation parameters (max_new_tokens, temperature, etc.)
        
        Returns:
            Generated text
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        if self.model is None:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Can clean up resources here
        pass
