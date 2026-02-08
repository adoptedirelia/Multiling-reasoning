def reasoner_prompt(question_en: str) -> str:
    return f"""
You are an advanced reasoning assistant. You will be provided with a question. Your task is to analyze the query step-by-step and provide a direct answer in English.

Instructions:
1.  Reasoning Process: Before answering, break down the problem logically. Analyze the constraints, perform necessary calculations, or outline your arguments. Enclose this entire thought process within <think> tags.
2.  Final Output: Provide only the final, concise result in English within <answer> tags.

Required Format:
<question>
[Question in English goes here]
</question>
<think>
[Detailed step-by-step logic and analysis in English goes here]
</think>
<answer>
[Final answer in English goes here]
</answer>

Input: {question_en}
Output:
"""


def mt2_standard_prompt(answer_en: str, target_lang: str) -> str:
    return f"""
You are a professional translator.
Translate the English answer into {target_lang}.
Return only the translated answer in <answer> tags.

Input:
{answer_en}

Output:
<answer>
[Answer in {target_lang}]
</answer>
"""


def mt2_context_prompt(x_l: str, x_en_hat: str, r_en_hat: str, y_en_hat: str, target_lang: str) -> str:
    return f"""
You are an advanced reasoning assistant. You will be provided with:
1) A question in {target_lang}
2) The same question in English (possibly corrupted)
3) English reasoning
4) An English answer

You must produce the best final answer in {target_lang}.
If there is conflict, prioritize faithfulness to the original {target_lang} question.

Required Format:
<answer>
[Final answer in {target_lang} goes here]
</answer>

Input (question in {target_lang}):
{x_l}

Input (question in English):
{x_en_hat}

English Thinking Process:
{r_en_hat}

English Answer:
{y_en_hat}

Output:
"""


def input_corruption_prompt(x_en: str, corruption_type: str) -> str:
    return f"""
You are corrupting an English question to simulate translation errors.
Create a corrupted English question that matches the given corruption type.

Corruption types:
- shift_intent: preserve topic but shift intent (e.g., verification -> explanation).

Return only the corrupted English question in <x_en_err> tags.
Do not include explanations, code, or any extra text.
Keep it to a single line.

Input question (English):
{x_en}

Corruption type:
{corruption_type}

Output:
<x_en_err>
[Corrupted question in English]
</x_en_err>
"""


def output_corruption_prompt(x_en: str, r_en: str, y_en: str, corruption_type: str) -> str:
    return f"""
You are corrupting English reasoning and answer outputs.
Generate corrupted versions that match the given corruption type.

Corruption types:
- assumption: introduce unstated assumptions in reasoning; answer follows that assumption.
- inconsistency: make reasoning and answer disagree.
- drift: reasoning is related but does not answer the question; answer drifts away.

Return only the corrupted reasoning and answer in <r_en> and <y_en> tags.
Do not include explanations, code, or any extra text.
Keep each field to 1-2 sentences.

Notes:
- The provided baseline reasoning may be empty.
- The baseline answer may contain multiple gold answers separated by " ||| ".
- Choose one gold answer and corrupt it.
- Ensure the corrupted reasoning and corrupted answer are consistent with the chosen corruption type.

Question (English):
{x_en}

Baseline reasoning:
{r_en}

Baseline answer:
{y_en}

Corruption type:
{corruption_type}

Output:
<r_en>
[Corrupted reasoning in English]
</r_en>
<y_en>
[Corrupted answer in English]
</y_en>
"""
