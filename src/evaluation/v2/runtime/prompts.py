from src.prompt import (
    END_TO_END_PROMPT,
    MT1_PROMPT,
    MT2_BASE_PROMPT,
    MT2_PROMPT,
    REASONING_PROMPT,
)


def reasoner_prompt(question_en: str) -> str:
    return REASONING_PROMPT.format(question=question_en)


def mt1_translate_prompt(x_l: str, src_lang: str = "the source language") -> str:
    del src_lang
    return MT1_PROMPT.format(question=x_l)


def mt2_standard_prompt(answer_en: str, target_lang: str) -> str:
    base = MT2_BASE_PROMPT.format(language=target_lang).rstrip()
    output_marker = "Output:"
    if base.endswith(output_marker):
        base = base[: -len(output_marker)].rstrip()
    return f"{base}\n\nInput: {answer_en}\nOutput:\n"


def mt2_context_prompt(x_l: str, x_en_hat: str, r_en_hat: str, y_en_hat: str, target_lang: str) -> str:
    return MT2_PROMPT.format(
        language=target_lang,
        question=x_l,
        English_Question=x_en_hat,
        English_Thinking_Process=r_en_hat,
        English_Answer=y_en_hat,
    )


def _mt2_ablation_prompt(
    target_lang: str,
    *,
    x_l: str | None = None,
    x_en_hat: str | None = None,
    r_en_hat: str | None = None,
    y_en_hat: str,
) -> str:
    lines = [
        f"You are an advanced reasoning assistant. You will be provided with selected context fields and an English answer. You need to answer the question in the {target_lang}. You need to answer the question in cultural context of the {target_lang}.",
        "",
        "Required Format:",
        "<answer>",
        f"[Final answer in {target_lang} goes here]",
        "</answer>",
        "",
    ]
    if x_l is not None:
        lines.append(f"Input:{x_l}")
    if x_en_hat is not None:
        lines.append(f"English Question:{x_en_hat}")
    if r_en_hat is not None:
        lines.append(f"English Thinking Process:{r_en_hat}")
    lines.append(f"English Answer:{y_en_hat}")
    lines.extend(["", "Output:"])
    return "\n".join(lines)


def mt2_answer_plus_source_question_prompt(x_l: str, y_en_hat: str, target_lang: str) -> str:
    return _mt2_ablation_prompt(target_lang, x_l=x_l, y_en_hat=y_en_hat)


def mt2_answer_plus_english_question_prompt(
    x_en_hat: str, y_en_hat: str, target_lang: str
) -> str:
    return _mt2_ablation_prompt(target_lang, x_en_hat=x_en_hat, y_en_hat=y_en_hat)


def mt2_answer_plus_reasoning_prompt(r_en_hat: str, y_en_hat: str, target_lang: str) -> str:
    return _mt2_ablation_prompt(target_lang, r_en_hat=r_en_hat, y_en_hat=y_en_hat)


def input_corruption_prompt(x_en: str, corruption_type: str) -> str:
    return f"""
Corrupt the English question to simulate translation noise.

Allowed corruption type:
- shift_intent: preserve topic but shift the user's intent.

Output only one <x_en_err>...</x_en_err> block.
Do not output explanations, notes, code, or extra tags.

Question:
{x_en}

Corruption type:
{corruption_type}

Corrupted Question:
<x_en_err>
"""


def output_corruption_prompt(x_en: str, r_en: str, y_en: str, corruption_type: str) -> str:
    return f"""
Corrupt the English reasoning and answer according to the requested type.
Return realistic errors, not random nonsense.

Corruption types:
- contradiction:
  Make the output contradict a known real-world fact relevant to the question.
  (entity-level, relation-level, or full-sentence contradiction are allowed.)
- invented:
  Introduce an unsupported, unverifiable claim presented as fact.
  Keep it plausible. Avoid obvious fantasy/mythical entities.
- subjective:
  Replace factual grounding with opinion, preference, or value judgment.
  The output should be non-universal and not objectively verifiable.

Output only these two blocks:
<r_en>...</r_en>
<y_en>...</y_en>

Do not output explanations, notes, code, comments, or any other tags.
Keep each block to 1-3 sentences.
Keep the topic aligned with the original question.

Question:
{x_en}

Baseline reasoning:
{r_en}

Baseline answer:
{y_en}

Corruption type:
{corruption_type}

Your Output Format:
<r_en>
[Corrupted English reasoning]
</r_en>
<y_en>
[Corrupted English answer]
</y_en>
"""


def direct_answer_prompt(question_l: str, question_lang: str) -> str:
    return END_TO_END_PROMPT.format(language=question_lang, question=question_l)
