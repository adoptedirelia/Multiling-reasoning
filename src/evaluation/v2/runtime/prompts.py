def reasoner_prompt(question_en: str) -> str:
    return f"""
You are an advanced reasoning assistant. You will be provided with a question in English.

Requirements:
1. Write reasoning in English inside <think>...</think>.
2. Keep reasoning concise: exactly 2 to 3 sentences.
3. Write the final answer in English inside <answer>...</answer>.
4. Output only these two tags and nothing else.

Input: {question_en}
Your Output Format:
<think>
[2-3 sentence reasoning in English]
</think>
<answer>
[Final answer in English]
</answer>
"""


def mt1_translate_prompt(x_l: str, src_lang: str = "the source language") -> str:
    return f"""
Translate the following text into English with <answer> ... </answer>.
Do NOT answer the question. Do NOT add facts. Do NOT explain. Do NOT transliterate words into Latin letters.
Preserve intent and sentence type (question stays a question).

{src_lang}:
{x_l}

English:
<answer>
"""


def mt2_standard_prompt(answer_en: str, target_lang: str) -> str:
    return f"""
Translate the English input into {target_lang} with <answer>...</answer>.

English:
{answer_en}

{target_lang}:
<answer>
"""


def mt2_context_prompt(x_l: str, x_en_hat: str, r_en_hat: str, y_en_hat: str, target_lang: str) -> str:
    return f"""
You are given:
1) Original question in {target_lang}
2) English translation of that question
3) English reasoning
4) English answer

Produce the best final answer in {target_lang}.
Prioritize faithfulness to the original {target_lang} question.
Output only one <answer>...</answer> block.

Question in {target_lang}:
{x_l}

Question in English:
{x_en_hat}

English reasoning:
{r_en_hat}

English answer:
{y_en_hat}

{target_lang} answer:
<answer>
"""


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
    return f"""
Answer the following question in {question_lang}.
Output only one <answer>...</answer> block.
Do not include explanation.

Question:
{question_l}

Answer in {question_lang}:
<answer>
"""
