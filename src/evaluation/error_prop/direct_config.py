import json
from dataclasses import dataclass, field

from .config import DatasetConfig, GenerationConfig, ModelEngineConfig, OutputConfig


@dataclass
class DirectEvalConfig:
    run_name: str
    dataset: DatasetConfig
    model: ModelEngineConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def load_direct_config(path: str) -> DirectEvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = DatasetConfig(**raw["dataset"])
    if dataset.dataset_type == "mkqa" and not dataset.mkqa_path:
        raise ValueError("dataset.mkqa_path is required for mkqa")
    if dataset.dataset_type == "global_piqa" and not dataset.hf_name:
        raise ValueError("dataset.hf_name is required for global_piqa")
    if dataset.dataset_type == "aya" and not dataset.hf_name:
        raise ValueError("dataset.hf_name is required for aya")

    model_raw = raw.get("model")
    if model_raw is None:
        raise ValueError("model is required")

    model = ModelEngineConfig(**model_raw)
    generation = GenerationConfig(**raw.get("generation", {}))
    outputs = OutputConfig(**raw.get("outputs", {}))

    return DirectEvalConfig(
        run_name=raw.get("run_name", "direct_eval"),
        dataset=dataset,
        model=model,
        generation=generation,
        outputs=outputs,
    )
