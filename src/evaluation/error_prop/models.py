from typing import List, Dict

from src.eval.engine import BaseEngine

from .prompts import (
    reasoner_prompt,
    mt2_context_prompt,
    mt2_standard_prompt,
    input_corruption_prompt,
    output_corruption_prompt,
)
from .text_utils import extract_answer, extract_reasoning, extract_tag


class ENReasoner:
    def __init__(self, engine: BaseEngine):
        self.engine = engine

    def run_batch(self, questions_en: List[str], max_new_tokens: int, temperature: float, top_p: float):
        prompts = [reasoner_prompt(q) for q in questions_en]
        outputs = self.engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out = []
        for raw in outputs:
            out.append(
                {
                    "raw": raw,
                    "reasoning": extract_reasoning(raw),
                    "answer": extract_answer(raw),
                }
            )
        return out


class MT2Standard:
    def __init__(self, engine: BaseEngine, target_lang: str):
        self.engine = engine
        self.target_lang = target_lang

    def run_batch(self, answers_en: List[str], max_new_tokens: int, temperature: float, top_p: float):
        prompts = [mt2_standard_prompt(a, self.target_lang) for a in answers_en]
        outputs = self.engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return [extract_answer(o) for o in outputs]


class MT2Context:
    def __init__(self, engine: BaseEngine, target_lang: str):
        self.engine = engine
        self.target_lang = target_lang

    def run_batch(
        self,
        x_l: List[str],
        x_en_hat: List[str],
        r_en_hat: List[str],
        y_en_hat: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        prompts = [
            mt2_context_prompt(xl, xe, r, y, self.target_lang)
            for xl, xe, r, y in zip(x_l, x_en_hat, r_en_hat, y_en_hat)
        ]
        outputs = self.engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return [extract_answer(o) for o in outputs]


def _invalid_llm_text(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    if "```" in text:
        return True
    if "<question>" in lowered or "<think>" in lowered or "<answer>" in lowered:
        return True
    if "explanation:" in lowered or "note:" in lowered:
        return True
    return False


def _fallback_shift_intent(x_en: str) -> str:
    return f"Explain {x_en}"


def _fallback_drop_constraints(x_en: str) -> str:
    return f"Provide a detailed response: {x_en}"


def _fallback_output_corruption(x_en: str, r_en: str, y_en: str, corruption_type: str) -> Dict[str, str]:
    if corruption_type == "assumption":
        r_err = f"{r_en} Assuming an unstated condition, the outcome changes."
        y_err = f"{y_en} (under that assumption)"
        return {"r_en_err": r_err, "y_en_err": y_err}
    if corruption_type == "inconsistency":
        r_err = f"{r_en} Therefore, the answer is 12."
        y_err = "15"
        return {"r_en_err": r_err, "y_en_err": y_err}
    if corruption_type == "drift":
        r_err = f"{r_en} This provides background context rather than a direct answer."
        y_err = "This topic has important historical significance."
        return {"r_en_err": r_err, "y_en_err": y_err}
    return {"r_en_err": r_en, "y_en_err": y_en}


class LLMInputCorruptor:
    def __init__(self, engine: BaseEngine):
        self.engine = engine

    def corrupt_batch(
        self,
        questions_en: List[str],
        corruption_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        base_prompts = [input_corruption_prompt(q, corruption_type) for q in questions_en]
        outputs = self.engine.generate_batch(
            base_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out = []
        retry_idxs = []
        for i, raw in enumerate(outputs):
            err = extract_tag(raw, "x_en_err")
            if not err or _invalid_llm_text(err):
                retry_idxs.append(i)
                out.append("")
            else:
                out.append(err)

        if retry_idxs:
            strict_prompts = [
                base_prompts[i] + "\nSTRICT: Output only <x_en_err> tags with one line."
                for i in retry_idxs
            ]
            retry_outs = self.engine.generate_batch(
                strict_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for idx, raw in zip(retry_idxs, retry_outs):
                err = extract_tag(raw, "x_en_err")
                if not err or _invalid_llm_text(err):
                    if corruption_type == "shift_intent":
                        out[idx] = _fallback_shift_intent(questions_en[idx])
                    else:
                        out[idx] = _fallback_drop_constraints(questions_en[idx])
                else:
                    out[idx] = err
        return out


class LLMOutputCorruptor:
    def __init__(self, engine: BaseEngine):
        self.engine = engine

    def corrupt_batch(
        self,
        questions_en: List[str],
        reasonings_en: List[str],
        answers_en: List[str],
        corruption_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Dict[str, str]]:
        base_prompts = [
            output_corruption_prompt(q, r, a, corruption_type)
            for q, r, a in zip(questions_en, reasonings_en, answers_en)
        ]
        outputs = self.engine.generate_batch(
            base_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        out = []
        retry_idxs = []
        for i, raw in enumerate(outputs):
            r_err = extract_tag(raw, "r_en")
            y_err = extract_tag(raw, "y_en")
            if _invalid_llm_text(r_err) or _invalid_llm_text(y_err):
                retry_idxs.append(i)
                out.append({"r_en_err": "", "y_en_err": ""})
            else:
                out.append({"r_en_err": r_err, "y_en_err": y_err})

        if retry_idxs:
            strict_prompts = [
                base_prompts[i]
                + "\nSTRICT: Output only <r_en> and <y_en> tags, 1-2 sentences each."
                for i in retry_idxs
            ]
            retry_outs = self.engine.generate_batch(
                strict_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for idx, raw in zip(retry_idxs, retry_outs):
                r_err = extract_tag(raw, "r_en")
                y_err = extract_tag(raw, "y_en")
                if _invalid_llm_text(r_err) or _invalid_llm_text(y_err):
                    out[idx] = _fallback_output_corruption(
                        questions_en[idx], reasonings_en[idx], answers_en[idx], corruption_type
                    )
                else:
                    out[idx] = {"r_en_err": r_err, "y_en_err": y_err}
        return out
