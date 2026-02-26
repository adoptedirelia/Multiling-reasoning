import re
from typing import Dict, List

from src.eval.engine import BaseEngine

from .prompts import (
    input_corruption_prompt,
    mt1_translate_prompt,
    mt2_context_prompt,
    mt2_standard_prompt,
    output_corruption_prompt,
    reasoner_prompt,
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
        for i, raw in enumerate(outputs):
            body = _strip_prompt_echo(raw, prompts[i])
            reasoning = extract_reasoning(body)
            answer = extract_answer(body)
            if _invalid_reasoner_pair(reasoning, answer):
                answer = _sanitize_answer_fallback(answer, questions_en[i])
                reasoning = ""
            out.append({"raw": raw, "reasoning": reasoning, "answer": answer})
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
        return [extract_answer(_strip_prompt_echo(o, prompts[i])) for i, o in enumerate(outputs)]


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
        return [extract_answer(_strip_prompt_echo(o, prompts[i])) for i, o in enumerate(outputs)]


class MT1Translator:
    def __init__(self, engine: BaseEngine, src_lang: str):
        self.engine = engine
        self.src_lang = src_lang

    def run_batch(self, x_l_list: List[str], max_new_tokens: int, temperature: float, top_p: float):
        prompts = [mt1_translate_prompt(xl, self.src_lang) for xl in x_l_list]
        outputs = self.engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        cleaned = []
        for i, raw in enumerate(outputs):
            answer_blocks = _extract_tag_all(raw, "answer")
            chosen = ""
            for cand in reversed(answer_blocks):
                cand = _clean_mt1_text(cand)
                if not _invalid_translation_text(cand):
                    chosen = cand
                    break
            if not chosen:
                body = _strip_prompt_echo(raw, prompts[i]).strip()
                fallback = _clean_mt1_text(body)
                chosen = fallback if fallback else ""
            cleaned.append(chosen)
        return cleaned


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
        for i, raw in enumerate(outputs):
            err = extract_tag(_strip_prompt_echo(raw, base_prompts[i]), "x_en_err")
            if not err or _invalid_llm_text(err):
                if corruption_type == "shift_intent":
                    out.append(_fallback_shift_intent(questions_en[i]))
                else:
                    out.append(_fallback_drop_constraints(questions_en[i]))
            else:
                out.append(err)
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
        for i, raw in enumerate(outputs):
            body = _strip_prompt_echo(raw, base_prompts[i])
            r_err = extract_tag(body, "r_en")
            y_err = extract_tag(body, "y_en")
            if _invalid_llm_text(r_err) or _invalid_llm_text(y_err):
                out.append(
                    _fallback_output_corruption(
                        questions_en[i], reasonings_en[i], answers_en[i], corruption_type
                    )
                )
            else:
                out.append({"r_en_err": r_err, "y_en_err": y_err})
        return out


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


def _strip_prompt_echo(raw: str, prompt: str) -> str:
    r = (raw or "")
    p = (prompt or "")
    if not p:
        return r
    if r.startswith(p):
        return r[len(p) :].lstrip()
    r_l = r.lstrip()
    if r_l.startswith(p):
        return r_l[len(p) :].lstrip()
    if p in r:
        return r.rsplit(p, 1)[-1].lstrip()
    return r


def _invalid_reasoner_pair(reasoning: str, answer: str) -> bool:
    if not answer:
        return True
    lower = answer.lower()
    if "<think>" in lower or "<question>" in lower or "</answer>" in lower:
        return True
    if "```" in answer:
        return True
    if len(answer) > 400 and "<answer>" not in answer.lower():
        return True
    return False


def _invalid_translation_text(text: str) -> bool:
    if not text:
        return True
    lower = text.lower()
    if "<question>" in lower or "<think>" in lower or "<answer>" in lower or "</answer>" in lower:
        return True
    if "```" in text:
        return True
    if lower.startswith("output:") or lower.startswith("input:"):
        return True
    if "translate the following text from" in lower:
        return True
    if "translate the following into english" in lower:
        return True
    if "translation:" in lower or "explanation:" in lower or "note:" in lower:
        return True
    if lower in {"[english translation]", "[corrupted english question]", "...", "{translation}", "[translation only]"}:
        return True
    if lower in {"[translation]"}:
        return True
    if lower.startswith("[english translation"):
        return True
    return False


def _extract_tag_all(text: str, tag: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [m.strip() for m in re.findall(pattern, t, re.DOTALL | re.IGNORECASE)]


def _clean_mt1_text(text: str) -> str:
    t = (text or "").replace("```", " ").strip()
    if not t:
        return ""
    lower = t.lower()
    markers = [
        "\ntranslation:",
        "\nexplanation:",
        "\nnote:",
        "\noutput:",
        "\ninput:",
        " translate the following text from",
    ]
    cuts = [lower.find(m) for m in markers if lower.find(m) != -1]
    if cuts:
        t = t[: min(cuts)].strip()
    close_idx = t.lower().find("</answer>")
    if close_idx != -1:
        t = t[:close_idx].strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _sanitize_answer_fallback(answer: str, question_en: str) -> str:
    if not answer:
        return question_en
    first_line = answer.strip().splitlines()[0].strip()
    if not first_line:
        return question_en
    return first_line


def _fallback_shift_intent(x_en: str) -> str:
    return f"Explain {x_en}"


def _fallback_drop_constraints(x_en: str) -> str:
    return f"Provide a detailed response: {x_en}"


def _fallback_output_corruption(x_en: str, r_en: str, y_en: str, corruption_type: str) -> Dict[str, str]:
    if corruption_type == "contradiction":
        r_err = f"{r_en} Historical records indicate a different factual outcome."
        y_err = "The accepted factual answer is different from this claim."
        return {"r_en_err": r_err, "y_en_err": y_err}
    if corruption_type == "invented":
        r_err = f"{r_en} Independent reports cite the undocumented Meridian Registry as confirmation."
        y_err = "According to the Meridian Registry, this is the confirmed answer."
        return {"r_en_err": r_err, "y_en_err": y_err}
    if corruption_type == "subjective":
        r_err = f"{r_en} In my opinion, the best interpretation is the most inspiring one."
        y_err = "I think the best answer is the most meaningful one."
        return {"r_en_err": r_err, "y_en_err": y_err}
    return {"r_en_err": r_en, "y_en_err": y_en}
