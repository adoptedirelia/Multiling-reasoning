import re
from typing import Tuple


def extract_answer(text: str) -> str:
    t = text.strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def extract_reasoning(text: str) -> str:
    t = text.strip()
    m = re.search(r"<think>(.*?)</think>", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_tag(text: str, tag: str) -> str:
    t = text.strip()
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""
