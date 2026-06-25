import gc
from typing import List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .base import BaseEngine


class MistralEngine(BaseEngine):
    """Mistral-Instruct engine via vLLM with proper chat template application."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str = "flash_attention_2",
        lora_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.lora_path = lora_path
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.llm_kwargs = kwargs

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_lora=self.lora_path is not None,
            **self.llm_kwargs,
        )

    def _apply_template(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        if self.model is None:
            self.load_model()

        formatted = self._apply_template(prompt, system_prompt)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        outputs = self.model.generate([formatted], sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip("\n")

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        if self.model is None:
            self.load_model()

        formatted_prompts = [self._apply_template(p, system_prompt) for p in prompts]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        lora_request = LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None
        outputs = self.model.generate(
            formatted_prompts, sampling_params=sampling_params, lora_request=lora_request
        )
        return [output.outputs[0].text.strip("\n") for output in outputs]

    def shutdown(self):
        model = self.model
        try:
            if model is not None:
                llm_engine = getattr(model, "llm_engine", None)
                if llm_engine is not None:
                    shutdown = getattr(llm_engine, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                close = getattr(model, "close", None)
                if callable(close):
                    close()
        except Exception:
            pass
        finally:
            self.model = None
            self.tokenizer = None

        try:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
