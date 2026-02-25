from typing import Optional, List, Union
from .base import BaseEngine
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class Qwen3Engine(BaseEngine):
    """Qwen3 model engine implementation"""
    
    def __init__(self, model_name: str, device_map: str = "auto", 
                 torch_dtype: str = "auto", attn_implementation: str = "flash_attention_2", lora_path: Optional[str] = None, **kwargs):
        """
        Initialize Qwen3 engine
        
        Args:
            model_name: Model name or path
            device_map: Device mapping
            torch_dtype: Torch data type
            attn_implementation: Attention implementation
        """
        super().__init__(model_name, **kwargs)
        # Keep these for API compatibility with existing config, but vLLM manages devices internally.
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.lora_path = lora_path
        # vLLM-specific options (can be passed via **kwargs)
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.llm_kwargs = kwargs
    
    def load_model(self):
        """Load Qwen3 model via vLLM."""
        # Note: vLLM handles dtype/device placement internally.
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_lora=True,
            **self.llm_kwargs,
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 max_new_tokens: int = 8192, temperature: float = 0.7, 
                 top_p: float = 0.9, **kwargs) -> str:
        """
        Generate text using Qwen3
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter
            top_p: Top-p sampling parameter
            **kwargs: Other generation parameters
        
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()

        # Build final textual prompt (this project already uses string templates in `prompt.py`)
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        outputs = self.model.generate([full_prompt], sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""

        text = outputs[0].outputs[0].text
        return text.strip("\n")

    def generate_batch(self, prompts: List[str], system_prompt: Optional[str] = None, 
                       max_new_tokens: int = 8192, temperature: float = 0.7, 
                       top_p: float = 0.9, **kwargs) -> List[str]:
        """
        Generate text using Qwen3
        
        Args:
            prompts: List of input prompts
            system_prompt: System prompt (optional)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for each prompt
            top_p: Top-p sampling parameter for each prompt
            **kwargs: Other generation parameters
        
        Returns:
            List of generated texts
        """
        if self.model is None:
            self.load_model()

        if system_prompt:
            prompts = [f"{system_prompt}\n\n{prompt}" for prompt in prompts]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        if self.lora_path:
            lora_request = LoRARequest('adapter',1,self.lora_path)
        else:
            lora_request = None

        outputs = self.model.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)

        return [output.outputs[0].text.strip("\n") for output in outputs]