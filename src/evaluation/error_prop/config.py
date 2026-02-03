import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.eval.config import ModelConfig


@dataclass
class ModelEngineConfig(ModelConfig):
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str = "mkqa"
    split: str = "train"
    langs: List[str] = field(default_factory=lambda: ["zh_cn", "ar", "ja", "vi", "mr", "am", "te"])
    max_examples: Optional[int] = None
    seed: int = 42
    path: Optional[str] = None


@dataclass
class GenerationConfig:
    translation_batch_size: int = 16
    translation_max_new_tokens: int = 128
    reasoning_batch_size: int = 8
    reasoning_max_new_tokens: int = 256
    reasoning_temperature: float = 0.0
    reasoning_top_p: float = 1.0
    mt2_mode: str = "translator"  # "translator" or "corrector"
    mt2_batch_size: int = 8
    mt2_max_new_tokens: int = 256


@dataclass
class OutputConfig:
    results_csv: str = "results/constraints_eval.csv"
    logs_dir: str = "logs"


@dataclass
class EvaluationConfig:
    run_name: str
    dataset: DatasetConfig
    models: Dict[str, ModelEngineConfig]
    constraints_file: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def _load_model_config(raw: Dict[str, Any]) -> ModelEngineConfig:
    return ModelEngineConfig(**raw)


def load_config(path: str) -> EvaluationConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = DatasetConfig(**raw.get("dataset", {}))
    generation = GenerationConfig(**raw.get("generation", {}))
    outputs = OutputConfig(**raw.get("outputs", {}))

    models_raw = raw.get("models", {})
    mt1_raw = models_raw.get("mt1") or models_raw.get("MT1")
    reasoner_raw = models_raw.get("reasoner_en") or models_raw.get("REASONER_EN")
    mt2_raw = models_raw.get("mt2") or models_raw.get("MT2")
    if mt1_raw is None or reasoner_raw is None or mt2_raw is None:
        raise ValueError(
            "Missing required models.mt1/models.reasoner_en/models.mt2 "
            "(or models.MT1/models.REASONER_EN/models.MT2) in config."
        )

    models = {
        "mt1": _load_model_config(mt1_raw),
        "reasoner_en": _load_model_config(reasoner_raw),
        "mt2": _load_model_config(mt2_raw),
    }

    constraints_file = raw.get("constraints_file")
    if not constraints_file:
        raise ValueError("Missing constraints_file in config.")

    run_name = raw.get("run_name", "constraints_eval")
    return EvaluationConfig(
        run_name=run_name,
        dataset=dataset,
        models=models,
        constraints_file=constraints_file,
        generation=generation,
        outputs=outputs,
    )
