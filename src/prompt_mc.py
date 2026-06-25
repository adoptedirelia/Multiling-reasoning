OPTION_LABELS = ["A", "B", "C", "D", "E"]
OPTION_FIELDS = ["option_a", "option_b", "option_c", "option_d", "option_e"]


def build_options_text(opts: dict) -> str:
    """Build formatted options block from an options dict, skipping empty/missing entries.
    e.g. 'A. foo\\nB. bar\\nC. baz\\nD. qux' (4 options)
    or   'A. foo\\nB. bar\\nC. baz\\nD. qux\\nE. quux' (5 options)
    """
    lines = []
    for label, field in zip(OPTION_LABELS, OPTION_FIELDS):
        text = opts.get(field, "")
        if text:
            lines.append(f"{label}. {text}")
    return "\n".join(lines)


def build_answer_hint(opts: dict) -> str:
    """Build dynamic answer hint like 'A, B, C, or D' / 'A, B, C, D, or E'."""
    present = []
    for label, field in zip(OPTION_LABELS, OPTION_FIELDS):
        if opts.get(field, ""):
            present.append(label)
    if len(present) <= 1:
        return ", ".join(present)
    return ", ".join(present[:-1]) + ", or " + present[-1]


def build_options_translation_format(opts: dict) -> str:
    """Build the required-format section for per-option translation output tags."""
    lines = []
    for label, field in zip(OPTION_LABELS, OPTION_FIELDS):
        if opts.get(field, ""):
            lines.append(
                f"<{field}_translation>\n"
                f"[English translation of option {label} goes here]\n"
                f"</{field}_translation>"
            )
    return "\n".join(lines)


MC_MT_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a multiple-choice question and the English thinking process and answer. You need to answer the question by selecting the correct option.

Required Format:
<think>
[Detailed step-by-step logic and analysis goes here]
</think>
<answer>
[{answer_hint}]
</answer>

Input: {question}
{options_text}
Output:
"""

MC_MT1_PROMPT = """
Translate the following multiple-choice question and all its options to English.

Required Format:
<question_translation>
[English translation of the question goes here]
</question_translation>
{options_translation_format}

Input:
{question}
{options_text}

Output:
"""

MC_REASONING_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a multiple-choice question. Your task is to analyze each option step-by-step and select the correct answer in English.

Instructions:
1.  Reasoning Process: Before answering, break down the problem logically. Evaluate each option and explain why it is correct or incorrect. Enclose this entire thought process within <think> tags.
2.  Final Output: Provide only the letter of the correct option ({answer_hint}) within <answer> tags.

Required Format:
<think>
[Detailed step-by-step logic and analysis of each option in English goes here]
</think>
<answer>
[{answer_hint}]
</answer>

Input: {question}
{options_text}
Output:
"""

MC_MT2_BASE_PROMPT = """
You are an advanced translation assistant. You will be provided with a sentence in English. You need to translate the sentence to {language}.

Required Format:
<answer>
[{answer_hint}]
</answer>

Input: {English_Answer}
Output:
"""

MC_MT2_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a multiple-choice question, the English question, the English thinking process and answer. You need to select the correct option for the question. You need to answer the question in the cultural context of the {language}.

Required Format:
<answer>
[{answer_hint}]
</answer>

Input: {question}
{options_text}
English Question: {English_Question}
English Thinking Process: {English_Thinking_Process}
English Answer: {English_Answer}

Output:
"""

MC_END_TO_END_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a multiple-choice question in {language}. Your task is to analyze each option step-by-step and select the correct answer in {language}.

Instructions:
1.  Reasoning Process: Before answering, break down the problem logically. Evaluate each option and explain why it is correct or incorrect. Enclose this entire thought process within <think> tags.
2.  Final Output: Provide only the letter of the correct option ({answer_hint}) within <answer> tags.

Required Format:
<think>
[Detailed step-by-step logic and analysis of each option in {language} goes here]
</think>
<answer>
[{answer_hint}]
</answer>

Input: {question}
{options_text}
Output:
"""

# ---------------------------------------------------------------------------
# Ablation study MC prompts (MT2 step variations for the prompting pipeline)
# ---------------------------------------------------------------------------


MC_ABLATION_ORIG_Q_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a multiple-choice question in {language} and the English answer. You need to select the correct option.

Required Format:
<answer>
[{answer_hint}]
</answer>

Input: {question}
{options_text}
English Answer: {English_Answer}

Output:
"""

MC_ABLATION_ENG_Q_PROMPT = """
You are an advanced reasoning assistant. You will be provided with an English multiple-choice question and the English answer. You need to select the correct option and provide the answer in {language}.

Required Format:
<answer>
[{answer_hint}]
</answer>

English Question: {English_Question}
English Answer: {English_Answer}

Output:
"""

MC_ABLATION_REASONING_PROMPT = """
You are an advanced reasoning assistant. You will be provided with the English thinking process and the English answer to a multiple-choice question. You need to select the correct option and provide the answer in {language}.

Required Format:
<answer>
[{answer_hint}]
</answer>

English Thinking Process: {English_Thinking_Process}
English Answer: {English_Answer}

Output:
"""
