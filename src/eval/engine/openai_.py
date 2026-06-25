import json
import logging
import os
import time
from typing import List, Optional

import requests

from .base import BaseEngine

LOGGER = logging.getLogger(__name__)


class OpenAIEngine(BaseEngine):
    """OpenAI chat completions engine."""

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1/chat/completions",
        timeout_s: int = 60,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.timeout_s = timeout_s
        # Fixed retry budget to avoid infinite loops.
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

    def load_model(self):
        # No local model to load; validate API key exists.
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {self.api_key_env}")
        self.model = True

    def _headers(self):
        key = os.environ.get(self.api_key_env)
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
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                # Retry only transient HTTP statuses.
                if resp.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(
                        f"Transient HTTP {resp.status_code}",
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as err:
                last_err = err
                retryable = True
                if isinstance(err, requests.exceptions.HTTPError):
                    status = getattr(err.response, "status_code", None)
                    retryable = status in {408, 409, 429, 500, 502, 503, 504}
                if (not retryable) or (attempt >= self.max_retries):
                    raise
                sleep_s = self.retry_backoff_s * (2 ** attempt)
                LOGGER.warning(
                    "OpenAI request failed (%s). Retrying %d/%d in %.1fs",
                    err,
                    attempt + 1,
                    self.max_retries,
                    sleep_s,
                )
                time.sleep(sleep_s)
        # Defensive fallback; should be unreachable due to raise in loop.
        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenAI request failed with unknown error")

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
