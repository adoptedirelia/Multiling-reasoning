MT_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a question and the English thinking process and answer. You need to answer the question.

Required Format:
<think>
[Detailed step-by-step logic and analysis goes here]
</think>
<answer>
[Final answer goes here]
</answer>

Input: {question}
Output:
"""

MT1_PROMPT = """
Translate the following question to English:
{question}

Required Format:
<translation>
[English translation of the question goes here]
</translation>

Output:
"""


REASONING_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a question. Your task is to analyze the query step-by-step and provide a direct answer in English.

Instructions:
1.  Reasoning Process: Before answering, break down the problem logically. Analyze the constraints, perform necessary calculations, or outline your arguments. Enclose this entire thought process within <think> tags.
2.  Final Output: Provide only the final, concise result in English within <answer> tags.

Required Format:
<think>
[Detailed step-by-step logic and analysis in English goes here]
</think>
<answer>
[Final answer in English goes here]
</answer>

Input: {question}
Output:

"""

MT2_BASE_PROMPT = """
You are an advanced translation assistant. You will be provided with a sentence in English. You need to translate the sentence to {language}.

Required Format:
<answer>
[Final answer in {language} goes here]
</answer>

Input: {English_Answer}
Output:
"""

MT2_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a question, the English question, the English thinking process and answer. You need to answer the question in the {language}. You need to answer the question in cultural context of the {language}.


Required Format:
<answer>
[Final answer in {language} goes here]
</answer>

Input:{question}
English Question:{English_Question}
English Thinking Process:{English_Thinking_Process}
English Answer:{English_Answer}

Output:
"""

END_TO_END_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a question in {language}. Your task is to analyze the query step-by-step and provide a direct answer in {language}.

Instructions:
1.  Reasoning Process: Before answering, break down the problem logically. Analyze the constraints, perform necessary calculations, or outline your arguments. Enclose this entire thought process within <think> tags.
2.  Final Output: Provide only the final, concise result in {language} within <answer> tags.

Required Format:
<think>
[Detailed step-by-step logic and analysis in {language} goes here]
</think>
<answer>
[Final answer in {language} goes here]
</answer>

Input: {question}
Output:
"""

MATH_REASONING_PROMPT = """
You are an advanced mathematics reasoning assistant. You will be provided with a math problem. Your task is to solve it step-by-step and present the final answer.

Instructions:
1.  Reasoning Process: Solve the problem step-by-step in English, showing all calculations and logical deductions. Enclose this entire process within <think> tags.
2.  Final Answer: Place only the final answer inside \\boxed{{}} immediately after the closing </think> tag.

Required Format:
<think>
[Detailed step-by-step solution in English goes here]
</think>
\\boxed{{[Final answer]}}

Input: {question}
Output:
"""

MATH_MT2_BASE_PROMPT = """
You are an advanced translation assistant. A math problem has been solved in English and the final answer is given below. Present this answer in {language}.

Required Format:
<answer>
\\boxed{{[Final answer]}}
</answer>

English Answer: {English_Answer}
Output:
"""

MATH_MT2_PROMPT = """
You are an advanced mathematics reasoning assistant. You will be provided with a math problem in {language}, along with its English translation and full English solution. Present the final answer in {language}.

Required Format:
<answer>
\\boxed{{[Final answer]}}
</answer>

Input: {question}
English Question: {English_Question}
English Thinking Process: {English_Thinking_Process}
English Answer: {English_Answer}

Output:
"""

MATH_END_TO_END_PROMPT = """
You are an advanced mathematics reasoning assistant. You will be provided with a math problem in {language}. Your task is to solve it step-by-step and present the final answer.

Instructions:
1.  Reasoning Process: Solve the problem step-by-step in {language}, showing all calculations and logical deductions. Enclose this entire process within <think> tags.
2.  Final Answer: Place only the final answer inside \\boxed{{}} immediately after the closing </think> tag.

Required Format:
<think>
[Detailed step-by-step solution in {language} goes here]
</think>
\\boxed{{[Final answer]}}

Input: {question}
Output:
"""

# ---------------------------------------------------------------------------
# Ablation study prompts (MT2 step variations for the prompting pipeline)
# ---------------------------------------------------------------------------

ABLATION_ORIG_Q_PROMPT = """
You are an advanced reasoning assistant. You will be provided with a question in {language} and the English answer. You need to answer the question in {language}.

Required Format:
<answer>
[Final answer in {language} goes here]
</answer>

Input: {question}
English Answer: {English_Answer}

Output:
"""

ABLATION_ENG_Q_PROMPT = """
You are an advanced reasoning assistant. You will be provided with the English question and the English answer. You need to provide the answer in {language}.

Required Format:
<answer>
[Final answer in {language} goes here]
</answer>

English Question: {English_Question}
English Answer: {English_Answer}

Output:
"""

ABLATION_REASONING_PROMPT = """
You are an advanced reasoning assistant. You will be provided with the English thinking process and the English answer to a question. You need to provide the answer in {language}.

Required Format:
<answer>
[Final answer in {language} goes here]
</answer>

English Thinking Process: {English_Thinking_Process}
English Answer: {English_Answer}

Output:
"""
