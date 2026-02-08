import json
import os
from typing import List, Optional

import requests

from .base import BaseEngine


class OpenAIEngine(BaseEngine):
    """OpenAI chat completions engine."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        timeout_s: int = 60,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_s = timeout_s

    def load_model(self):
        # No local model to load; validate API key exists.
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")
        self.model = True

    def _headers(self):
        key = os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        if self.model is None:
            self.load_model()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        resp = requests.post(
            self.base_url,
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> List[str]:
        # Simple sequential implementation for API calls.
        return [
            self.generate(
                prompt=p,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            for p in prompts
        ]
