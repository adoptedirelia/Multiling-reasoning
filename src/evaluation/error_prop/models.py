import re
from dataclasses import dataclass
from typing import List

from src.eval.engine import BaseEngine

from .prompts import build_mt2_corrector_prompt, build_reasoning_prompt, build_translate_prompt


def _extract_answer(text: str) -> str:
    t = text.strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    if "Response:" in t:
        return t.split("Response:")[-1].strip()
    return t


def _extract_reasoning(text: str) -> str:
    t = text.strip()
    reasoning_match = re.search(r"<think>(.*?)</think>", t, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return ""




@dataclass
class ReasonerOutput:
    answer: str
    reasoning: str
    raw_output: str


class LLMTranslator:
    def __init__(self, engine: BaseEngine, src_lang: str, tgt_lang: str, default_max_new_tokens: int = 128):
        self.engine = engine
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.default_max_new_tokens = default_max_new_tokens

    def batched_translate(self, texts: List[str], batch_size: int = 8, max_new_tokens: int = None) -> List[str]:
        outputs: List[str] = []
        use_max_new_tokens = max_new_tokens or self.default_max_new_tokens
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompts = [build_translate_prompt(t, self.src_lang, self.tgt_lang) for t in batch]
            chunk = self.engine.generate_batch(
                prompts,
                max_new_tokens=use_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            outputs.extend([_extract_answer(c) for c in chunk])
        return outputs


class LLMReasoner:
    def __init__(
        self,
        engine: BaseEngine,
        default_max_new_tokens: int = 256,
        default_temperature: float = 0.0,
        default_top_p: float = 1.0,
    ):
        self.engine = engine
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p

    def answer_batch(
        self,
        questions: List[str],
        batch_size: int = 8,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> List[str]:
        rich = self.reason_answer_batch(
            questions=questions,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return [x.answer for x in rich]

    def reason_answer_batch(
        self,
        questions: List[str],
        batch_size: int = 8,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> List[ReasonerOutput]:
        outputs: List[str] = []
        use_max_new_tokens = max_new_tokens or self.default_max_new_tokens
        use_temperature = self.default_temperature if temperature is None else temperature
        use_top_p = self.default_top_p if top_p is None else top_p

        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            prompts = [build_reasoning_prompt(q) for q in batch]
            chunk = self.engine.generate_batch(
                prompts,
                max_new_tokens=use_max_new_tokens,
                temperature=use_temperature,
                top_p=use_top_p,
            )
            outputs.extend(chunk)
        return [
            ReasonerOutput(
                answer=_extract_answer(c),
                reasoning=_extract_reasoning(c),
                raw_output=c.strip(),
            )
            for c in outputs
        ]


class Corrector:
    def __init__(self, engine: BaseEngine, target_lang: str, default_max_new_tokens: int = 256):
        self.engine = engine
        self.target_lang = target_lang
        self.default_max_new_tokens = default_max_new_tokens

    def produce_batch(
        self,
        *,
        q_l: List[str],
        q_en: List[str],
        y_en: List[str],
        reason_en: List[str],
        batch_size: int = 8,
        max_new_tokens: int = None,
    ) -> List[str]:
        outputs: List[str] = []
        use_max_new_tokens = max_new_tokens or self.default_max_new_tokens
        for i in range(0, len(q_l), batch_size):
            q_l_b = q_l[i : i + batch_size]
            q_en_b = q_en[i : i + batch_size]
            y_en_b = y_en[i : i + batch_size]
            reason_b = reason_en[i : i + batch_size]
            prompts = [
                build_mt2_corrector_prompt(
                    q_l=q_l_i,
                    q_en=q_en_i,
                    y_en=y_en_i,
                    reasoning_en=reason_i,
                    target_lang=self.target_lang,
                )
                for q_l_i, q_en_i, y_en_i, reason_i in zip(q_l_b, q_en_b, y_en_b, reason_b)
            ]
            chunk = self.engine.generate_batch(
                prompts,
                max_new_tokens=use_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            outputs.extend([_extract_answer(c) for c in chunk])
        return outputs
