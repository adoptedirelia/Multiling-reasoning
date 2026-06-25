import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelEngineConfig:
    model_type: str
    model_name: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dataset_type: str
    langs: List[str]
    max_examples: int = 100
    seed: int = 42
    mkqa_path: str = ""
    hf_name: Optional[str] = None
    hf_split: str = "test"
    hf_configs: Optional[Dict[str, str]] = None


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
class CascadeConfig:
    modes: List[str] = field(default_factory=lambda: ["standard", "context"])


@dataclass
class EvalConfig:
    methods: List[str] = field(default_factory=lambda: ["f1"])
    prediction_field: str = "prediction"
    slice_field: str = "slice"
    lang_field: str = "lang"
    example_id_field: str = "example_id"
    bertscore_model_type: str = "xlm-roberta-large"
    bertscore_batch_size: int = 16
    bertscore_rescale_with_baseline: bool = False
    win_judge_model_name: str = "gpt-4.1-mini"
    win_judge_api_key_env: str = "OPENAI_API_KEY"
    win_judge_timeout_s: int = 120
    win_max_new_tokens: int = 8
    win_temperature: float = 0.0
    win_top_p: float = 1.0
    win_write_judgments: bool = False
    win_judgments_jsonl: str = ""


@dataclass
class OutputConfig:
    corruption_jsonl: str = "results/v2/corruptions.jsonl"
    predictions_jsonl: str = "results/v2/predictions.jsonl"
    metrics_json: str = "results/v2/metrics.json"
    logs_dir: str = "logs"


@dataclass
class V2Config:
    run_name: str
    dataset: DatasetConfig
    models: Dict[str, ModelEngineConfig]
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def _load_model(raw: Dict[str, Any]) -> ModelEngineConfig:
    return ModelEngineConfig(**raw)


def load_config(path: str) -> V2Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = DatasetConfig(**raw["dataset"])
    if dataset.dataset_type == "mkqa" and not dataset.mkqa_path:
        raise ValueError("dataset.mkqa_path is required for mkqa")
    if dataset.dataset_type in {"aya", "global_piqa", "blend"} and not dataset.hf_name:
        raise ValueError("dataset.hf_name is required for aya/global_piqa/blend")

    models_raw = raw.get("models", {})
    reasoner_raw = models_raw.get("reasoner_en")
    mt2_raw = models_raw.get("mt2")
    mt1_raw = models_raw.get("mt1") or reasoner_raw
    corr_raw = models_raw.get("corruption_llm") or reasoner_raw
    if reasoner_raw is None or mt2_raw is None:
        raise ValueError("models.reasoner_en and models.mt2 are required")

    models = {
        "reasoner_en": _load_model(reasoner_raw),
        "mt2": _load_model(mt2_raw),
        "mt1": _load_model(mt1_raw),
        "corruption_llm": _load_model(corr_raw),
    }

    return V2Config(
        run_name=raw.get("run_name", "v2_run"),
        dataset=dataset,
        models=models,
        corruption=CorruptionConfig(**raw.get("corruption", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        cascade=CascadeConfig(**raw.get("cascade", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        outputs=OutputConfig(**raw.get("outputs", {})),
    )
