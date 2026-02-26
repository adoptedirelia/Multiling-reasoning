import re


_ANY_TAG_RE = re.compile(r"</?[a-zA-Z_][a-zA-Z0-9_]*>")


def _extract_tag_tolerant(text: str, tag: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    strict = re.findall(rf"<{tag}>(.*?)</{tag}>", t, re.DOTALL | re.IGNORECASE)
    if strict:
        return strict[-1].strip()

    open_pat = re.compile(rf"<{tag}>", re.IGNORECASE)
    starts = list(open_pat.finditer(t))
    if starts:
        start = starts[-1].end()
        remainder = t[start:]
        next_tag = re.search(r"</?[a-zA-Z_][a-zA-Z0-9_]*>", remainder)
        chunk = remainder[: next_tag.start()] if next_tag else remainder
        return chunk.strip()

    return ""


def _best_effort_line(text: str) -> str:
    t = (text or "").replace("```", " ").strip()
    if not t:
        return ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    bad = ("output:", "input:", "note:", "explanation:", "translation:")
    for ln in lines:
        low = ln.lower()
        if low.startswith(bad):
            continue
        if _ANY_TAG_RE.search(ln):
            continue
        return ln
    return lines[0]


def extract_answer(text: str) -> str:
    tagged = _extract_tag_tolerant(text, "answer")
    if tagged:
        return tagged
    line = _best_effort_line(text)
    return line if line else (text or "").strip()


def extract_reasoning(text: str) -> str:
    return _extract_tag_tolerant(text, "think")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_tag(text: str, tag: str) -> str:
    return _extract_tag_tolerant(text, tag)
