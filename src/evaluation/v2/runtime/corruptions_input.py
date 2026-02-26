import random
from typing import List, Optional

import spacy


_NLP = spacy.load("en_core_web_sm")


def _remove_token(text: str, token_idx: int) -> str:
    doc = _NLP(text)
    tokens = [t.text_with_ws for t in doc]
    if 0 <= token_idx < len(tokens):
        tokens[token_idx] = ""
    return "".join(tokens).strip()


def _replace_span(text: str, start: int, end: int, repl: str) -> str:
    return text[:start] + repl + text[end:]


def entity_swap_error(text: str, rng: random.Random) -> Optional[str]:
    doc = _NLP(text)
    ents = [e for e in doc.ents if e.text.strip()]
    if len(ents) >= 2:
        e1, e2 = rng.sample(ents, 2)
        if e1.start_char > e2.start_char:
            e1, e2 = e2, e1
        out = text
        out = _replace_span(out, e2.start_char, e2.end_char, e1.text)
        out = _replace_span(out, e1.start_char, e1.end_char, e2.text)
        return out
    nums = [t for t in doc if t.like_num]
    if len(nums) >= 2:
        t1, t2 = rng.sample(nums, 2)
        if t1.idx > t2.idx:
            t1, t2 = t2, t1
        out = text
        out = _replace_span(out, t2.idx, t2.idx + len(t2.text), t1.text)
        out = _replace_span(out, t1.idx, t1.idx + len(t1.text), t2.text)
        return out
    tokens = [t for t in doc if not t.is_space]
    if len(tokens) >= 4:
        i = rng.randint(1, len(tokens) - 2)
        a, b = tokens[i], tokens[i + 1]
        out_tokens = [t.text_with_ws for t in doc]
        out_tokens[a.i], out_tokens[b.i] = out_tokens[b.i], out_tokens[a.i]
        out = "".join(out_tokens).strip()
        if out and out != text:
            return out
    return None


def omission_error(text: str, rng: random.Random, max_words: int = 2) -> Optional[str]:
    doc = _NLP(text)
    candidates = [t for t in doc if t.pos_ in {"ADJ", "ADV"}]
    if candidates:
        t = rng.choice(candidates)
        return _remove_token(text, t.i)
    preps = [t for t in doc if t.dep_ == "prep"]
    if preps:
        t = rng.choice(preps)
        span = doc[t.left_edge.i : t.right_edge.i + 1]
        if len([tok for tok in span]) <= max_words:
            out = text[: span.start_char] + text[span.end_char :]
            return out.strip()
    if max_words > 0:
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if tokens:
            for n in (2, 1):
                if n > max_words:
                    continue
                if len(tokens) < n:
                    continue
                idxs = {t.i for t in tokens[:n]}
                out_tokens = [t.text_with_ws if t.i not in idxs else "" for t in doc]
                out = "".join(out_tokens).strip()
                if out and out != text:
                    return out
    return None


def generate_input_errors(text: str, rng: random.Random, omission_max_words: int = 2) -> List[tuple]:
    variants = []
    om = omission_error(text, rng, max_words=omission_max_words)
    if om and om != text:
        variants.append(("omission", om))
    ent = entity_swap_error(text, rng)
    if ent and ent != text:
        variants.append(("entity_swap", ent))
    return variants
