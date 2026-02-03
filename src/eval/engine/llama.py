from typing import List, Optional

from vllm import LLM, SamplingParams

from .base import BaseEngine


class LlamaEngine(BaseEngine):
    """Llama model engine implementation via vLLM."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        self.llm_kwargs = kwargs

    def load_model(self):
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.llm_kwargs,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        if self.model is None:
            self.load_model()

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        outputs = self.model.generate([full_prompt], sampling_params=sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip("\n")

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
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
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        return [output.outputs[0].text.strip("\n") for output in outputs]
