import re
import string
from collections import Counter


MIXED_SEGMENTATION_LANGS = {"zh_cn", "zh_hk", "zh_tw", "ja", "th", "km"}
MIXED_SEGMENTATION_PREFIXES = {
    "zh",
    "zho",
    "cmn",
    "yue",
    "ja",
    "jpn",
    "th",
    "tha",
    "km",
    "khm",
}
MIXED_SEGMENTATION_SCRIPTS = {"hans", "hant", "jpan", "thai", "khmr"}
LANGUAGE_CODE_ALIASES = {
    "eng": "en",
    "spa": "es",
    "vie": "vi",
    "deu": "de",
    "ger": "de",
    "arb": "ar",
    "ary": "ar",
    "arz": "ar",
    "ars": "ar",
    "aeb": "ar",
    "acm": "ar",
    "acq": "ar",
    "afb": "ar",
    "apc": "ar",
    "arq": "ar",
    "bsk": "ar",
    "pes": "ar",
    "urd": "ar",
    "nld": "nl",
    "swe": "sv",
    "dan": "da",
    "nob": "no",
    "nno": "no",
    "fra": "fr",
    "por": "pt",
    "ita": "it",
    "fin": "fi",
    "hun": "hu",
}

ARTICLE_REGEX_BY_LANG = {
    "en": r"\b(a|an|the)\b",
    "es": r"\b(un|una|unos|unas|el|la|los|las)\b",
    "vi": r"\b(của|là|cái|chiếc|những)\b",
    "de": r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
    "ar": r"\sال^|ال",
    "nl": r"\b(de|het|een|des|der|den)\b",
    "sv": r"\b(en|ett)\b",
    "da": r"\b(en|et)\b",
    "no": r"\b(en|et|ei)\b",
    "fr": r"\b(le|la|l'|les|du|de|d'|des|un|une|des)",
    "pt": r"\b(o|a|os|as|um|uma|uns|umas)\b",
    "it": r"\b(il|lo|la|l'|i|gli|le|del|dello|della|dell'|dei|degli|degl'|delle|un'|uno|una|un)",
    "fi": r"\b(se|yks|yksi)\b",
    "hu": r"\b(a|az|egy)\b",
}


def whitespace_tokenize(text: str):
    return text.split()


def _canonicalize_lang(lang: str) -> str:
    return (lang or "").strip().lower().replace("-", "_")


def _lang_variants(lang: str):
    canonical = _canonicalize_lang(lang)
    if not canonical:
        return []

    parts = [p for p in canonical.split("_") if p]
    variants = [canonical]
    for part in parts:
        if part not in variants:
            variants.append(part)

    root = parts[0] if parts else canonical
    alias = LANGUAGE_CODE_ALIASES.get(root)
    if alias and alias not in variants:
        variants.append(alias)
    return variants


def _is_mixed_segmentation_char(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x3040 <= code <= 0x309F  # Hiragana
        or 0x30A0 <= code <= 0x30FF  # Katakana
        or 0x31F0 <= code <= 0x31FF  # Katakana Phonetic Extensions
        or 0x0E00 <= code <= 0x0E7F  # Thai
        or 0x1780 <= code <= 0x17FF  # Khmer
    )


def _contains_mixed_segmentation_chars(text: str) -> bool:
    return any(_is_mixed_segmentation_char(ch) for ch in text or "")


def _should_use_mixed_segmentation(lang: str, text: str) -> bool:
    variants = _lang_variants(lang)
    if any(v in MIXED_SEGMENTATION_LANGS for v in variants):
        return True
    if any(v in MIXED_SEGMENTATION_PREFIXES for v in variants):
        return True
    if any(v in MIXED_SEGMENTATION_SCRIPTS for v in variants):
        return True
    return _contains_mixed_segmentation_chars(text)


def mixed_segmentation(text: str):
    segs_out = []
    temp_str = ""
    for char in text:
        if char.isspace():
            if temp_str:
                segs_out.extend(whitespace_tokenize(temp_str))
                temp_str = ""
            continue
        if _is_mixed_segmentation_char(char):
            if temp_str:
                segs_out.extend(whitespace_tokenize(temp_str))
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char
    if temp_str != "":
        segs_out.extend(whitespace_tokenize(temp_str))
    return segs_out


def normalize_answer_by_language(s: str, lang: str) -> str:
    def remove_articles(text: str, lng: str):
        for variant in _lang_variants(lng):
            article_regex = ARTICLE_REGEX_BY_LANG.get(variant)
            if article_regex:
                return re.sub(article_regex, " ", text)
        return text

    def white_space_fix(text: str, lng: str):
        if _should_use_mixed_segmentation(lng, text):
            tokens = mixed_segmentation(text)
        else:
            tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower()), lang), lang)


def calculate_em(prediction: str, gold_answer: str, language: str):
    norm_pred = normalize_answer_by_language(prediction, language)
    norm_answer = normalize_answer_by_language(gold_answer, language)
    return int(norm_pred == norm_answer)


def calculate_f1(prediction: str, gold_answer: str, language: str):
    gold_toks = normalize_answer_by_language(gold_answer, language).split() if gold_answer else []
    pred_toks = normalize_answer_by_language(prediction, language).split() if prediction else []
    common = Counter(gold_toks) & Counter(pred_toks)
    num_common = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_common == 0:
        return 0.0

    recall = 1.0 * num_common / len(gold_toks)
    precision = 1.0 * num_common / len(pred_toks)
    return (2.0 * precision * recall) / (precision + recall)


def compute_max_score_over_answers(metric_fn, prediction: str, ground_truths, language: str):
    assert len(ground_truths) > 0, "Gold truth answers list should never be empty."
    scores_by_answer = [metric_fn(prediction, ground_truth, language) for ground_truth in ground_truths]
    return max(scores_by_answer)
