import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.eval.config import ModelConfig


@dataclass
class ModelEngineConfig(ModelConfig):
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    mkqa_path: str
    langs: List[str] = field(default_factory=lambda: ["zh_cn", "ar", "vi", "ja"])
    max_examples: int = 100
    seed: int = 42


@dataclass
class CorruptionConfig:
    num_variants_per_error: int = 1
    omission_max_words: int = 2
    require_all_errors: bool = True
    max_output_variants_per_type: int = 6
    input_jsonl: Optional[str] = None
    output_jsonl: Optional[str] = None


@dataclass
class GenerationConfig:
    mt1_max_new_tokens: int = 256
    mt1_temperature: float = 0.0
    mt1_top_p: float = 1.0
    mt2_max_new_tokens: int = 256
    mt2_temperature: float = 0.0
    mt2_top_p: float = 1.0
    corruption_max_new_tokens: int = 256
    corruption_temperature: float = 0.3
    corruption_top_p: float = 0.9


@dataclass
class OutputConfig:
    predictions_dir: str = "results/error_sim/preds"
    metrics_dir: str = "results/error_sim/metrics"
    logs_dir: str = "logs"


@dataclass
class ErrorSimConfig:
    run_name: str
    dataset: DatasetConfig
    models: Dict[str, ModelEngineConfig]
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def _load_model_config(raw: Dict[str, Any]) -> ModelEngineConfig:
    return ModelEngineConfig(**raw)


def load_config(path: str) -> ErrorSimConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = DatasetConfig(**raw["dataset"])
    corruption = CorruptionConfig(**raw.get("corruption", {}))
    generation = GenerationConfig(**raw.get("generation", {}))
    outputs = OutputConfig(**raw.get("outputs", {}))

    models_raw = raw.get("models", {})
    reasoner_raw = models_raw.get("reasoner_en") or models_raw.get("REASONER_EN")
    mt2_raw = models_raw.get("mt2") or models_raw.get("MT2")
    corr_raw = (
        models_raw.get("corruption_llm")
        or models_raw.get("CORRUPTION_LLM")
        or models_raw.get("corruptor_llm")
        or models_raw.get("CORRUPTOR_LLM")
    )
    if reasoner_raw is None or mt2_raw is None:
        raise ValueError("models.reasoner_en and models.mt2 are required")

    models = {
        "reasoner_en": _load_model_config(reasoner_raw),
        "mt2": _load_model_config(mt2_raw),
    }
    if corr_raw is not None:
        models["corruption_llm"] = _load_model_config(corr_raw)

    return ErrorSimConfig(
        run_name=raw.get("run_name", "error_sim"),
        dataset=dataset,
        models=models,
        corruption=corruption,
        generation=generation,
        outputs=outputs,
    )
