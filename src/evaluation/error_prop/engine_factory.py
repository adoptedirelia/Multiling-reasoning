from src.eval.engine import BaseEngine, LlamaEngine, Qwen3Engine

from .config import ModelEngineConfig


def create_engine(model_cfg: ModelEngineConfig) -> BaseEngine:
    model_type = model_cfg.model_type.lower()
    if model_type == "qwen3":
        return Qwen3Engine(
            model_name=model_cfg.model_name,
            device_map=model_cfg.device_map,
            torch_dtype=model_cfg.torch_dtype,
            attn_implementation=model_cfg.attn_implementation,
            **model_cfg.engine_kwargs,
        )
    if model_type == "llama":
        return LlamaEngine(
            model_name=model_cfg.model_name,
            device_map=model_cfg.device_map,
            torch_dtype=model_cfg.torch_dtype,
            attn_implementation=model_cfg.attn_implementation,
            **model_cfg.engine_kwargs,
        )

    raise ValueError(f"Unsupported model_type={model_cfg.model_type}. Supported: qwen3, llama.")
