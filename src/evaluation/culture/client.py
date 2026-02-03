import json
import os
import re
from typing import Dict, Optional

import requests


def _parse_score(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\b(10|[1-9])\b", text)
    if not match:
        return None
    return int(match.group(1))


def _parse_scores(text: str) -> Dict[str, int]:
    text = text.strip()
    try:
        data = json.loads(text)
        culture = int(data.get("culture"))
        correctness = int(data.get("correctness"))
        if 1 <= culture <= 10 and 1 <= correctness <= 10:
            return {"culture": culture, "correctness": correctness}
    except Exception:
        pass

    # Fallback: try to extract two integers in order.
    nums = re.findall(r"\b(10|[1-9])\b", text)
    if len(nums) >= 2:
        return {"culture": int(nums[0]), "correctness": int(nums[1])}
    raise RuntimeError(f"Failed to parse scores from response: {text}")


def _clamp_scores(scores: Dict[str, int]) -> Dict[str, int]:
    def clamp(v: int) -> int:
        return 1 if v < 1 else 10 if v > 10 else v

    return {
        "culture": clamp(int(scores["culture"])),
        "correctness": clamp(int(scores["correctness"])),
    }


class LLMClient:
    def __init__(self, provider: str, model: str, api_key_env: str, timeout_s: int):
        self.provider = provider.lower()
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s

    def _get_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")
        return key

    def score(self, prompt: str, temperature: float = 0.0, max_output_tokens: int = 128) -> Dict[str, int]:
        if self.provider == "openai":
            return _clamp_scores(self._score_openai(prompt, temperature, max_output_tokens))
        if self.provider == "gemini":
            return _clamp_scores(self._score_gemini(prompt, temperature, max_output_tokens))
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _score_openai(self, prompt: str, temperature: float, max_output_tokens: int) -> Dict[str, int]:
        key = self._get_key()
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict grader. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return _parse_scores(text)

    def _score_gemini(self, prompt: str, temperature: float, max_output_tokens: int) -> Dict[str, int]:
        key = self._get_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return _parse_scores(text)
