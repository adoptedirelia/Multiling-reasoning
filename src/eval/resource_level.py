"""
Resource-level classification (high / mid / low) for each multilingual dataset.

Scores come from `resource_score_log_pages` in std_cxt_e2e_language_features.csv.

Thresholds:
  high : score >= 16.0
  mid  : 12.0 <= score < 16.0
  low  : score < 12.0, or no score available in the CSV
"""

from __future__ import annotations

import os
import pandas as pd

_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "../../std_cxt_e2e_language_features.csv")

HIGH_THRESHOLD = 16.0
MID_THRESHOLD  = 12.0


# ── Load CSV once at import time ──────────────────────────────────────────────

def _load_lookups() -> tuple[dict, dict, dict]:
    df = pd.read_csv(_CSV_PATH)
    flores_to_score: dict[str, float | None] = {}
    code2_to_score:  dict[str, float | None] = {}
    code3_to_score:  dict[str, float | None] = {}  # resource_lookup_code fallback

    for _, row in df.iterrows():
        flores = str(row["language"]).strip().lower()
        code2  = str(row["language_code"]).strip().lower()
        code3  = str(row["resource_lookup_code"]).strip().lower()
        raw    = row["resource_score_log_pages"]
        score  = float(raw) if str(raw).replace(".", "", 1).isdigit() else None

        flores_to_score[flores] = score
        if code2 not in code2_to_score or (code2_to_score[code2] is None and score is not None):
            code2_to_score[code2] = score
        if code3 not in code3_to_score or (code3_to_score[code3] is None and score is not None):
            code3_to_score[code3] = score

    return flores_to_score, code2_to_score, code3_to_score


_FLORES_TO_SCORE, _CODE2_TO_SCORE, _CODE3_TO_SCORE = _load_lookups()

# BLEnD uses US/GB country codes for English; alias them to eng_latn's score
_FLORES_TO_SCORE["us"] = _FLORES_TO_SCORE.get("eng_latn")
_FLORES_TO_SCORE["gb"] = _FLORES_TO_SCORE.get("eng_latn")


def _tier(score: float | None) -> str:
    if score is None:
        return "low"
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MID_THRESHOLD:
        return "mid"
    return "low"


def _by_code2(lang_map: dict[str, str]) -> dict[str, str]:
    """Map {2-letter-code: EnglishName} → {EnglishName: tier}."""
    return {name: _tier(_CODE2_TO_SCORE.get(code.lower()))
            for code, name in lang_map.items()}


def _flores_score(code: str) -> float | None:
    """Look up score by flores code, falling back to 3-letter language prefix."""
    key = code.lower()
    score = _FLORES_TO_SCORE.get(key)
    if score is not None:
        return score
    # Fallback: try the 3-letter prefix (e.g. "zho_hans" → "zho")
    prefix = key.split("_")[0]
    score = _FLORES_TO_SCORE.get(prefix)
    if score is not None:
        return score
    # Final fallback: resource_lookup_code index
    return _CODE3_TO_SCORE.get(prefix)


def _by_flores(lang_map: dict[str, str]) -> dict[str, str]:
    """Map {flores-code: EnglishName} → {EnglishName: tier}. Lowercases the key."""
    return {name: _tier(_flores_score(code)) for code, name in lang_map.items()}


def _by_code3(lang_map: dict[str, str]) -> dict[str, str]:
    """Map {resource_lookup_code (3-letter): EnglishName} → {EnglishName: tier}."""
    return {name: _tier(_CODE3_TO_SCORE.get(code.lower()))
            for code, name in lang_map.items()}


# ── Per-dataset language maps ─────────────────────────────────────────────────

_GLOBAL_MMLU = {
    "am": "Amharic",    "ar": "Arabic",      "bn": "Bengali",    "cs": "Czech",
    "de": "German",     "el": "Greek",       "en": "English",    "fil": "Filipino",
    "fr": "French",     "ha": "Hausa",       "he": "Hebrew",     "hi": "Hindi",
    "ig": "Igbo",       "id": "Indonesian",  "it": "Italian",    "ja": "Japanese",
    "ky": "Kyrgyz",     "ko": "Korean",      "lt": "Lithuanian", "mg": "Malagasy",
    "ms": "Malay",      "ne": "Nepali",      "nl": "Dutch",      "ny": "Chichewa",
    "fa": "Persian",    "pl": "Polish",      "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian",    "si": "Sinhala",     "sn": "Shona",      "so": "Somali",
    "es": "Spanish",    "sr": "Serbian",     "sw": "Swahili",    "sv": "Swedish",
    "te": "Telugu",     "tr": "Turkish",     "uk": "Ukrainian",  "vi": "Vietnamese",
    "yo": "Yoruba",     "zh": "Chinese",
}

_BELEBELE = {
    "acm_Arab": "Mesopotamian Arabic",
    "afr_Latn": "Afrikaans",
    "als_Latn": "Tosk Albanian",
    "amh_Ethi": "Amharic",
    "apc_Arab": "North Levantine Arabic",
    "arb_Arab": "Modern Standard Arabic",
    "arb_Latn": "Modern Standard Arabic (Romanized)",
    "ars_Arab": "Najdi Arabic",
    "ary_arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic",
    "asm_Beng": "Assamese",
    "azj_Latn": "North Azerbaijani",
    "bam_Latn": "Bambara",
    "ben_Beng": "Bengali",
    "ben_Latn": "Bengali (Romanized)",
    "bod_Tibt": "Standard Tibetan",
    "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan",
    "ceb_Latn": "Cebuano",
    "ces_Latn": "Czech",
    "ckb_Arab": "Central Kurdish",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "eng_Latn": "English",
    "est_Latn": "Estonian",
    "eus_Latn": "Basque",
    "fin_Latn": "Finnish",
    "fra_Latn": "French",
    "fuv_Latn": "Nigerian Fulfulde",
    "gaz_Latn": "West Central Oromo",
    "grn_Latn": "Guarani",
    "guj_Gujr": "Gujarati",
    "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "hin_Latn": "Hindi (Romanized)",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo",
    "ilo_Latn": "Ilocano",
    "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic",
    "ita_Latn": "Italian",
    "jav_Latn": "Javanese",
    "jpn_Jpan": "Japanese",
    "kac_Latn": "Jingpho",
    "kan_Knda": "Kannada",
    "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh",
    "kea_Latn": "Kabuverdianu",
    "khk_Cyrl": "Halh Mongolian",
    "khm_Khmr": "Khmer",
    "kin_Latn": "Kinyarwanda",
    "kir_Cyrl": "Kyrgyz",
    "kor_Hang": "Korean",
    "lao_Laoo": "Lao",
    "lin_Latn": "Lingala",
    "lit_Latn": "Lithuanian",
    "lug_Latn": "Ganda",
    "luo_Latn": "Luo",
    "lvs_Latn": "Standard Latvian",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mkd_Cyrl": "Macedonian",
    "mlt_Latn": "Maltese",
    "mri_Latn": "Maori",
    "mya_Mymr": "Burmese",
    "nld_Latn": "Dutch",
    "nob_Latn": "Norwegian Bokmål",
    "npi_Deva": "Nepali",
    "npi_Latn": "Nepali (Romanized)",
    "nso_Latn": "Northern Sotho",
    "nya_Latn": "Nyanja",
    "ory_Orya": "Odia",
    "pan_Guru": "Eastern Panjabi",
    "pbt_Arab": "Southern Pashto",
    "pes_Arab": "Western Persian",
    "plt_Latn": "Plateau Malagasy",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "shn_Mymr": "Shan",
    "sin_Latn": "Sinhala (Romanized)",
    "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "sna_Latn": "Shona",
    "snd_Arab": "Sindhi",
    "som_Latn": "Somali",
    "sot_Latn": "Southern Sotho",
    "spa_Latn": "Spanish",
    "srp_Cyrl": "Serbian",
    "ssw_Latn": "Swati",
    "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish",
    "swh_Latn": "Swahili",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik",
    "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai",
    "tir_Ethi": "Tigrinya",
    "tsn_Latn": "Tswana",
    "tso_Latn": "Tsonga",
    "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian",
    "urd_Arab": "Urdu",
    "urd_Latn": "Urdu (Romanized)",
    "uzn_Latn": "Northern Uzbek",
    "vie_Latn": "Vietnamese",
    "war_Latn": "Waray",
    "wol_Latn": "Wolof",
    "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "zsm_Latn": "Standard Malay",
    "zul_Latn": "Zulu",
}

_MCSQA = {
    "en": "English", "ja": "Japanese", "zh": "Chinese",  "de": "German",
    "pt": "Portuguese", "nl": "Dutch",   "fr": "French",  "ru": "Russian",
}

_MMATH = {
    "ar": "Arabic",  "en": "English",    "es": "Spanish", "fr": "French",
    "ja": "Japanese", "ko": "Korean",    "pt": "Portuguese", "th": "Thai",
    "vi": "Vietnamese", "zh": "Chinese",
}

_PIQA = {
    "acm_arab":       "Mesopotamian Arabic",
    "acq_arab":       "Ta'izzi-Adeni Arabic",
    "aeb_arab":       "Tunisian Arabic",
    "afb_arab":       "Gulf Arabic",
    "als_latn":       "Tosk Albanian",
    "amh_ethi":       "Amharic",
    "apc_arab_jord":  "North Levantine Arabic (Jordan)",
    "apc_arab_leba":  "North Levantine Arabic (Lebanon)",
    "apc_arab_pale":  "North Levantine Arabic (Palestine)",
    "apc_arab_syri":  "North Levantine Arabic (Syria)",
    "arb_arab":       "Modern Standard Arabic",
    "arq_arab":       "Algerian Arabic",
    "ars_arab":       "Najdi Arabic",
    "ary_arab":       "Moroccan Arabic",
    "arz_arab":       "Egyptian Arabic",
    "asm_beng":       "Assamese",
    "azj_latn":       "North Azerbaijani",
    "bam_latn":       "Bambara",
    "bel_cyrl":       "Belarusian",
    "ben_beng":       "Bengali",
    "ben_latn":       "Bengali (Romanized)",
    "bho_deva":       "Bhojpuri",
    "bos_latn":       "Bosnian",
    "bsk_arab":       "Burushaski",
    "bul_cyrl":       "Bulgarian",
    "cat_latn":       "Catalan",
    "ces_latn":       "Czech",
    "ckb_arab":       "Central Kurdish",
    "ckm_latn":       "Kumzari",
    "cmn_hans":       "Chinese (Simplified)",
    "cmn_hant":       "Chinese (Traditional)",
    "deu_latn":       "German",
    "dhd_deva":       "Dhundari",
    "ekk_latn":       "Estonian",
    "ekp_latn":       "Ekpeye",
    "ell_grek":       "Greek",
    "eng_latn":       "English",
    "fao_latn":       "Faroese",
    "fin_latn":       "Finnish",
    "fra_latn_cana":  "French (Canada)",
    "fra_latn_fran":  "French (France)",
    "glg_latn":       "Galician",
    "guj_gujr":       "Gujarati",
    "hau_latn":       "Hausa",
    "haw_latn":       "Hawaiian",
    "heb_hebr":       "Hebrew",
    "hin_deva":       "Hindi",
    "hrv_latn":       "Croatian",
    "hun_latn":       "Hungarian",
    "hye_armn":       "Armenian",
    "ibo_latn":       "Igbo",
    "idu_latn":       "Idoma",
    "ind_latn":       "Indonesian",
    "isl_latn":       "Icelandic",
    "iso_latn":       "Isoko",
    "ita_latn":       "Italian",
    "jav_latn":       "Javanese",
    "jpn_jpan":       "Japanese",
    "kan_knda":       "Kannada",
    "kat_geor":       "Georgian",
    "kaz_cyrl":       "Kazakh",
    "kin_latn":       "Kinyarwanda",
    "kir_cyrl":       "Kyrgyz",
    "kor_hang":       "Korean",
    "lin_latn":       "Lingala",
    "lit_latn":       "Lithuanian",
    "luo_latn":       "Luo",
    "mal_mlym":       "Malayalam",
    "mar_deva":       "Marathi",
    "mkd_cyrl":       "Macedonian",
    "mni_beng":       "Meitei (Bengali script)",
    "mni_mtei":       "Meitei (Meitei script)",
    "nag_latn":       "Nagamese",
    "nld_latn":       "Dutch",
    "nno_latn":       "Norwegian Nynorsk",
    "nob_latn":       "Norwegian Bokmål",
    "npi_deva":       "Nepali",
    "pan_guru":       "Eastern Panjabi",
    "pcm_latn":       "Nigerian Pidgin",
    "pes_arab":       "Western Persian",
    "pol_latn":       "Polish",
    "por_latn_braz":  "Portuguese (Brazil)",
    "por_latn_port":  "Portuguese (Portugal)",
    "ron_latn":       "Romanian",
    "rus_cyrl":       "Russian",
    "rwr_deva":       "Marwari",
    "sin_sinh":       "Sinhala",
    "slk_latn":       "Slovak",
    "slk_latn_sari":  "Slovak (SARI)",
    "slv_latn":       "Slovenian",
    "slv_latn_cerk":  "Slovenian (CERK)",
    "snd_arab":       "Sindhi",
    "snd_deva":       "Sindhi (Devanagari)",
    "spa_latn_mexi":  "Spanish (Mexico)",
    "spa_latn_peru":  "Spanish (Peru)",
    "spa_latn_spai":  "Spanish (Spain)",
    "srp_cyrl":       "Serbian (Cyrillic)",
    "srp_latn":       "Serbian (Latin)",
    "swe_latn":       "Swedish",
    "swh_latn":       "Swahili",
    "tam_taml":       "Tamil",
    "tel_telu":       "Telugu",
    "tgl_latn":       "Tagalog",
    "tha_thai":       "Thai",
    "tur_latn":       "Turkish",
    "uig_arab":       "Uyghur",
    "ukr_cyrl":       "Ukrainian",
    "urd_arab":       "Urdu",
    "urd_latn":       "Urdu (Romanized)",
    "urh_latn":       "Urhobo",
    "uzn_latn":       "Northern Uzbek",
    "vie_latn":       "Vietnamese",
    "yor_latn":       "Yoruba",
    "yue_hant":       "Cantonese",
    "zsm_latn":       "Standard Malay",
    "zul_latn":       "Zulu",
}

_MGSM = {
    "bn": "Bengali", "de": "German",   "en": "English", "es": "Spanish",
    "fr": "French",  "ja": "Japanese", "ru": "Russian", "sw": "Swahili",
    "te": "Telugu",  "th": "Thai",     "zh": "Chinese",
}

# Aya — 3-letter resource_lookup_codes
_AYA = {
    "zho": "Chinese",
    "ara": "Arabic",   # dataset uses "arb" (ISO 639-3 MSA), but CSV lookup_code is "ara"
    "jpn": "Japanese",
    "vie": "Vietnamese",
    "mar": "Marathi",
    "amh": "Amharic",
    "tel": "Telugu",
}

# MKQA — 26 languages; codes are the `language` column values in the CSV
_MKQA = {
    "eng":   "English",   # CSV language col is "eng", not "en"
    "ar":    "Arabic",
    "da":    "Danish",
    "de":    "German",
    "es":    "Spanish",
    "fi":    "Finnish",
    "fr":    "French",
    "he":    "Hebrew",
    "hu":    "Hungarian",
    "it":    "Italian",
    "ja":    "Japanese",
    "km":    "Khmer",
    "ko":    "Korean",
    "ms":    "Malay",
    "nl":    "Dutch",
    "no":    "Norwegian",
    "pl":    "Polish",
    "pt":    "Portuguese",
    "ru":    "Russian",
    "sv":    "Swedish",
    "th":    "Thai",
    "tr":    "Turkish",
    "vi":    "Vietnamese",
    "zh_cn": "Chinese (Simplified)",
    "zh_hk": "Chinese (HK)",
    "zh_tw": "Chinese (Traditional)",
}

# BLEnD — 16 country splits merged to 14 unique language entries:
#   KR + KP  → Korean (same language, different country)
#   ES + MX  → Spanish (same language, different country)
#   US + GB  → English (US) / English (GB) (kept separate as distinct cultural contexts)
_BLEND = {
    "us": "English (US)",   # US  → United States
    "gb": "English (GB)",   # GB  → United Kingdom
    "cn": "Chinese",        # CN  → China
    "es": "Spanish",        # ES+MX → Spain / Mexico
    "id": "Indonesian",     # ID  → Indonesia
    "kr": "Korean",         # KR+KP → South / North Korea
    "gr": "Greek",          # GR  → Greece
    "ir": "Persian",        # IR  → Iran
    "dz": "Arabic",         # DZ  → Algeria
    "az": "Azerbaijani",    # AZ  → Azerbaijan
    "jb": "Javanese",       # JB  → Java, Indonesia
    "as": "Assamese",       # AS  → Assam, India
    "ng": "Hausa",          # NG  → Nigeria
    "et": "Amharic",        # ET  → Ethiopia
}

# ── Public API ────────────────────────────────────────────────────────────────

RESOURCE_LEVELS: dict[str, dict[str, str]] = {
    "global_mmlu": _by_code2(_GLOBAL_MMLU),
    "belebele":    _by_flores(_BELEBELE),
    "mcsqa":       _by_code2(_MCSQA),
    "mmath":       _by_code2(_MMATH),
    "piqa":        _by_flores(_PIQA),
    "mgsm":        _by_code2(_MGSM),
    "aya":         _by_code3(_AYA),
    "mkqa":        _by_flores(_MKQA),
    "blend":       _by_flores(_BLEND),
}


def get_resource_level(dataset: str, language: str) -> str:
    """Return 'high', 'mid', or 'low' for a language in a given dataset.

    dataset  : one of 'global_mmlu', 'belebele', 'mcsqa', 'mmath', 'piqa', 'mgsm'
    language : the English language name as used in result JSON files
               (e.g. 'Chinese', 'Arabic', 'Swahili')
    """
    key = dataset.lower().replace("-", "_").replace(" ", "_")
    return RESOURCE_LEVELS.get(key, {}).get(language, "low")


def get_languages_by_tier(dataset: str) -> dict[str, list[str]]:
    """Return {'high': [...], 'mid': [...], 'low': [...]} for a dataset."""
    key = dataset.lower().replace("-", "_").replace(" ", "_")
    levels = RESOURCE_LEVELS.get(key, {})
    result: dict[str, list[str]] = {"high": [], "mid": [], "low": []}
    for lang, tier in sorted(levels.items()):
        result[tier].append(lang)
    return result


# ── Quick summary when run directly ──────────────────────────────────────────

if __name__ == "__main__":
    for ds in ("global_mmlu", "belebele", "mcsqa", "mmath", "piqa", "mgsm",
               "aya", "mkqa", "blend"):
        tiers = get_languages_by_tier(ds)
        counts = {t: len(v) for t, v in tiers.items()}
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  |  {counts}")
        for tier in ("high", "mid", "low"):
            langs = tiers[tier]
            if langs:
                print(f"  [{tier:4s}] {', '.join(langs)}")
