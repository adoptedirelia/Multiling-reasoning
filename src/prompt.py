MT1_SYSTEM_PROMPT = """
"""


MT1_PROMT = """
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

"""

MT2_SYSTEM_PROMPT = """
"""

MT2_PROMT = """
You are an advanced reasoning assistant. You will be provided with a question and the English thinking process and answer. You need to answer the question in the {language}.

Required Format:
<think>
[Detailed step-by-step logic and analysis in {language} goes here]
</think>
<answer>
[Final answer in {language} goes here]
</answer>

Input: {question}
English Thinking Process:{English_Thinking_Process}

Output:
"""

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