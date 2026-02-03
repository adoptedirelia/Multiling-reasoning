import json
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProviderConfig:
    provider: str  # "openai" or "gemini"
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_output_tokens: int = 128


@dataclass
class CultureEvalConfig:
    input_path: str
    output_path: str
    providers: List[ProviderConfig]
    request_timeout_s: int = 60
    max_samples: Optional[int] = None


def load_config(path: str) -> CultureEvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    providers = [ProviderConfig(**p) for p in raw["providers"]]
    return CultureEvalConfig(
        input_path=raw["input_path"],
        output_path=raw["output_path"],
        providers=providers,
        request_timeout_s=raw.get("request_timeout_s", 60),
        max_samples=raw.get("max_samples"),
    )
