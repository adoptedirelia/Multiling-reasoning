import random
import re
from typing import Dict, List, Optional, Tuple

import spacy


_NLP = spacy.load("en_core_web_sm")


def _replace_span(text: str, start: int, end: int, repl: str) -> str:
    return text[:start] + repl + text[end:]


def build_entity_pool(texts: List[str]) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {}
    for t in texts:
        doc = _NLP(t)
        for ent in doc.ents:
            pool.setdefault(ent.label_, []).append(ent.text)
    # de-dup
    for k in list(pool.keys()):
        pool[k] = sorted(set(pool[k]))
    if not pool:
        pool["FALLBACK"] = ["ENTITY_A", "ENTITY_B", "ENTITY_C"]
    return pool


def entity_error(text: str, pool: Dict[str, List[str]], rng: random.Random) -> Optional[str]:
    doc = _NLP(text)
    ents = [e for e in doc.ents if e.label_ in pool and pool[e.label_]]
    if not ents:
        return None
    e = rng.choice(ents)
    candidates = [c for c in pool[e.label_] if c != e.text]
    if not candidates:
        return None
    repl = rng.choice(candidates)
    return _replace_span(text, e.start_char, e.end_char, repl)


def relation_error(text: str, rng: random.Random) -> Optional[str]:
    doc = _NLP(text)
    # find simple subject-verb-object pattern
    verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if not verb:
        return None
    subj = next((t for t in verb.lefts if t.dep_ in {"nsubj", "nsubjpass"}), None)
    obj = next((t for t in verb.rights if t.dep_ in {"dobj", "pobj", "attr"}), None)
    if not subj or not obj:
        return None
    subj_span = doc[subj.left_edge.i : subj.right_edge.i + 1]
    obj_span = doc[obj.left_edge.i : obj.right_edge.i + 1]
    # swap subject/object with passive transform
    swapped = f"{obj_span.text} was {verb.lemma_} by {subj_span.text}"
    # replace full sentence with swapped relation (syntactic error)
    return swapped


def fallback_entity_error(text: str) -> Optional[str]:
    doc = _NLP(text)
    for tok in doc:
        if tok.like_num:
            return _replace_span(text, tok.idx, tok.idx + len(tok.text), "9999")
    for tok in doc:
        if tok.pos_ == "PROPN":
            return _replace_span(text, tok.idx, tok.idx + len(tok.text), "ENTITY_A")
    if text:
        return f"{text} (ENTITY_A)"
    return None


def fallback_relation_error(text: str) -> Optional[str]:
    # simple predicate flip
    for marker in (" is ", " was ", " are ", " were "):
        if marker in text:
            return text.replace(marker, f"{marker.strip()} not ", 1)
    if text:
        return f"{text} (not)"
    return None


def corrupt_text_with_type(
    text: str, err_type: str, pool: Dict[str, List[str]], rng: random.Random
) -> Optional[str]:
    if err_type == "entity":
        out = entity_error(text, pool, rng)
        if not out:
            out = fallback_entity_error(text)
        return out
    if err_type == "relation":
        out = relation_error(text, rng)
        if not out:
            out = fallback_relation_error(text)
        return out
    return None


def generate_output_errors(text: str, pool: Dict[str, List[str]], rng: random.Random) -> List[Tuple[str, str]]:
    variants = []
    for err_type in ("entity", "relation"):
        out = corrupt_text_with_type(text, err_type, pool, rng)
        if out and out != text:
            variants.append((err_type, out))
    return variants


def generate_output_error_variants(
    text: str,
    err_type: str,
    pool: Dict[str, List[str]],
    rng: random.Random,
    max_variants: int,
) -> List[str]:
    variants: List[str] = []
    seen = set()
    attempts = 0
    max_attempts = max(10, max_variants * 5)
    while len(variants) < max_variants and attempts < max_attempts:
        attempts += 1
        out = corrupt_text_with_type(text, err_type, pool, rng)
        if not out or out == text or out in seen:
            continue
        seen.add(out)
        variants.append(out)

    # forced fallbacks to guarantee count
    idx = 1
    while len(variants) < max_variants:
        if err_type == "entity":
            forced = f"{text} (ENTITY_{idx})"
        else:
            forced = f"{text} (REL_{idx})"
        if forced not in seen:
            seen.add(forced)
            variants.append(forced)
        idx += 1

    return variants


def apply_repeated_corruption(
    text: str,
    err_type: str,
    pool: Dict[str, List[str]],
    rng: random.Random,
    max_errors: int,
) -> str:
    out = text
    attempts = 0
    max_attempts = max(5, max_errors * 4)
    applied = 0
    while applied < max_errors and attempts < max_attempts:
        attempts += 1
        mutated = corrupt_text_with_type(out, err_type, pool, rng)
        if mutated and mutated != out:
            out = mutated
            applied += 1
    return out
