LANG_NAMES = {
    "vi": "Vietnamese",
    "ar": "Arabic",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "ja": "Japanese",
    "mr": "Marathi",
    "te": "Telugu",
    "am": "Amharic",
    "amrabic": "Amharic",
    "en": "English",
}


def language_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


def build_translate_prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    return (
        f"You are a professional translator.\n"
        f"Translate the text from {language_name(src_lang)} to {language_name(tgt_lang)}.\n"
        "Return the translation wrapped in <answer>...</answer> tags only.\n\n"
        "Required Format:\n"
        "<answer>\n"
        f"[Translated text in {language_name(tgt_lang)}]\n"
        "</answer>\n\n"
        f"Input text:\n{text}\n\n"
        "Output:"
    )


def build_reasoning_prompt(question_en: str) -> str:
    return """
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

Input: {question}
Output:
""".format(question=question_en)


def build_mt2_corrector_prompt(
    *,
    q_l: str,
    q_en: str,
    y_en: str,
    reasoning_en: str,
    target_lang: str,
) -> str:
    lang = language_name(target_lang)
    return f"""
You are an advanced reasoning assistant. You will be provided with:
1) A question in {lang}
2) The same question in English
3) English reasoning
4) An English answer

You need to produce the best final answer in {lang}.
If there is conflict between inputs, prioritize faithfulness to the original {lang} question.

Required Format:
<think>
[Brief reasoning in {lang} goes here]
</think>
<answer>
[Final answer in {lang} goes here]
</answer>

Input (question in {lang}):
{q_l}

Input (question in English):
{q_en}

English Thinking Process:
{reasoning_en}

English Answer:
{y_en}

Output:
"""
