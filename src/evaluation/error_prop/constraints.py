import re
from typing import Callable, Dict, List


def _strip_ws(text: str) -> str:
    return text.strip()


def _has_only_numeric(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    return re.fullmatch(r"[+-]?\d[\d,]*(\.\d+)?", t) is not None


def _single_line(text: str) -> bool:
    return ("\n" not in text) and ("\r" not in text)


def _long_single_line(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    return _single_line(t) and len(t) >= 50


def _one_or_two_sentences(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    punct = re.findall(r"[.!?。！？]", t)
    if not punct:
        return True
    return 1 <= len(punct) <= 2


def _at_most_five_words(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    return len(re.findall(r"\S+", t)) <= 5


def _at_most_twenty_chars(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    return len(t) <= 20


def _at_least_fifty_chars(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    return len(t) >= 50


def _brief_then_three_reasons(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(paragraphs) != 4:
        return False
    # First paragraph should be brief (one line, <= 120 chars)
    if "\n" in paragraphs[0] or len(paragraphs[0]) > 120:
        return False
    # Remaining three paragraphs are reasons; require non-empty
    return all(bool(p) for p in paragraphs[1:])


def _exactly_two_paragraphs(text: str) -> bool:
    t = _strip_ws(text)
    if not t:
        return False
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    return len(paragraphs) == 2


CONSTRAINT_REGISTRY: Dict[str, Callable[[str], bool]] = {
    "numeric_only": _has_only_numeric,
    "single_line": _single_line,
    "one_two_sentences": _one_or_two_sentences,
    "five_words": _at_most_five_words,
    "max_chars_20": _at_most_twenty_chars,
    "long_single_line": _long_single_line,
    "min_chars_50": _at_least_fifty_chars,
    "brief_then_three_reasons": _brief_then_three_reasons,
    "exactly_two_paragraphs": _exactly_two_paragraphs,
}


def check_constraint(text: str, constraint_id: str) -> bool:
    fn = CONSTRAINT_REGISTRY.get(constraint_id)
    if fn is None:
        raise ValueError(f"Unknown constraint id: {constraint_id}")
    return fn(text)


def batch_constraint_accuracy(preds: List[str], constraint_id: str) -> float:
    if not preds:
        return 0.0
    ok = [check_constraint(p, constraint_id) for p in preds]
    return sum(ok) / len(ok)
