def build_answer_eval_prompt(question: str, answer: str) -> str:
    return f"""
You are evaluating a QA system response.

Question:
{question}

Response:
{answer}

Provide two scores on a 1-10 scale (integers only; do NOT use 0) to evaluate the response:
1) cultural relevance
2) correctness

Return only valid JSON in this exact format:
{{"culture": <int 1-10>, "correctness": <int 1-10>}}
"""


def build_ground_truth_eval_prompt(question: str, ground_truth: str) -> str:
    return f"""
You are evaluating a QA system response.

Question:
{question}

Response:
{ground_truth}

Provide two scores on a 1-10 scale (integers only; do NOT use 0) to evaluate the response:
1) cultural relevance
2) correctness

Return only valid JSON in this exact format:
{{"culture": <int 1-10>, "correctness": <int 1-10>}}
"""


def build_answer_with_ground_truth_eval_prompt(question: str, ground_truth: str, answer: str) -> str:
    return f"""
You are evaluating a QA system response with access to the ground-truth reference.

Question:
{question}

Ground truth:
{ground_truth}

Response:
{answer}

Provide two scores on a 1-10 scale (integers only; do NOT use 0) to evaluate the response:
1) cultural relevance
2) correctness

Return only valid JSON in this exact format:
{{"culture": <int 1-10>, "correctness": <int 1-10>}}
"""
