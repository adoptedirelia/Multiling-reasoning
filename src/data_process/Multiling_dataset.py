import json
import os
import pandas as pd

class PIQA_Dataset:
    """PIQA — 2-option physical intuition MC across 110 languages.
    Language key is the human-readable language name derived from the
    FLORES-200 filename stem.  Pass languages=[...] with name strings
    (e.g. 'Chinese (Simplified)', 'English') to load a subset.
    """

    LANGUAGE_MAP = {
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

    LABEL_TO_LETTER = {0: "A", 1: "B"}

    def __init__(self, dataset_path, save_path, languages=None, max_samples=None):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.max_samples = max_samples

        # Build {lang_name: filename} for every .tsv present in the directory.
        # Unknown codes fall back to the stem itself as the name.
        all_stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(dataset_path)
            if f.endswith(".tsv")
        )
        self.lang_name_to_file = {
            self.LANGUAGE_MAP.get(stem, stem): f"{stem}.tsv"
            for stem in all_stems
        }

        if languages is not None:
            lang_set = set(languages)
            self.lang_name_to_file = {
                k: v for k, v in self.lang_name_to_file.items() if k in lang_set
            }

        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "PIQA.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)


        with open(os.path.join(save_path, "PIQA_test.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

    def load_dataset(self):
        result = {}
        for lang_name, file_name in self.lang_name_to_file.items():
            print(f"Loading PIQA for {lang_name}...")
            data_path = os.path.join(self.dataset_path, file_name)
            df = pd.read_csv(data_path, sep="\t")

            result[lang_name] = []
            for row in df.itertuples(index=True):
                if self.max_samples and len(result[lang_name]) >= self.max_samples:
                    break
                option_a = row.solution0
                option_b = row.solution1
                answer_letter = self.LABEL_TO_LETTER.get(int(row.label), "A")
                answer_text = option_a if row.label == 0 else option_b

                result[lang_name].append({
                    "example_id": row.example_id,
                    "question": row.prompt,
                    "option_a": option_a,
                    "option_b": option_b,
                    "answer": answer_letter,
                    "answer_text": answer_text,
                })
        return result

class MKQA_Dataset:
    def __init__(self, dataset_path, save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.language_map = {
            "zh_cn": "Chinese",
            "ar": "Arabic",
            "ja": "Japanese",
            "vi": "Vietnamese",
        }
        self.dataset = self.load_dataset()
        with open(os.path.join(save_path, "MKQA.json"), "w") as f:
            json.dump(self.dataset, f, indent=4)

        eval_dataset = {}
        train_dataset = {}
        ratio = 0.8
        for language,data in self.dataset.items():
            train_data = data[500:]
            eval_data = data[:500]
            train_dataset[language] = train_data
            eval_dataset[language] = eval_data
        with open(os.path.join(save_path, "MKQA_train.json"), "w") as f:
            json.dump(train_dataset, f, indent=4)
        with open(os.path.join(save_path, "MKQA_eval.json"), "w") as f:
            json.dump(eval_dataset, f, indent=4)
    def load_dataset(self):
        res = {}
        with open(self.dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                english_question = data['query']
                example_id = data['example_id']
                for language,answer in data['answers'].items():
                    if language not in self.language_map.keys():
                        continue
                    language_name = self.language_map[language]
                    if language_name not in res:
                        res[language_name] = []
                    query = data['queries'][language]

                    res[language_name].append({
                        'english_question': english_question,
                        'question': query,
                        'example_id': example_id,
                        'answer': [ans['text'] for ans in answer]
                    })

        return res

class Aya_Dataset:
    def __init__(self, dataset_path, save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.language_map = {
            "Chinese": "zho",
            "Arabic": "ara",
            "Japanese": "jpn",
            "Vietnamese": "vie",
            "Marathi": "mar",
            "Amharic": "amh",
            "Telugu": "tel",
            "Simplified Chinese": "zho",
        }
        self.language_code_map = {
            "zho": "Chinese",
            "arb": "Arabic",
            "jpn": "Japanese",
            "vie": "Vietnamese",
            "mar": "Marathi",
            "amh": "Amharic",
            "tel": "Telugu",
        }
        self.dataset = self.load_dataset()

        train_dataset = {}
        test_dataset = {}
        for language,data in self.dataset['train'].items():
            train_data = data[500:]
            test_data = data[:500]
            train_dataset[language] = train_data
            test_dataset[language] = test_data

        with open(os.path.join(save_path, "Aya_train.json"), "w") as f:
            json.dump(train_dataset, f, indent=4)
        

        with open(os.path.join(save_path, "Aya_test.json"), "w") as f:
            json.dump(test_dataset, f, indent=4)
        

    def load_dataset(self):
        from datasets import load_dataset

        res = {'train': {}, 'test': {}}
        aya_dataset = load_dataset("CohereLabs/aya_dataset")

        for item in aya_dataset['train']:

            question = item['inputs']
            answer = item['targets']
            language = item['language']
            language_code = item['language_code']
            if language == 'Arabic':
                print(language_code)
            if language_code not in self.language_code_map.keys():
                continue
            # language_name = self.language_map[language]
            language_name = self.language_code_map[language_code]
            if language_name not in res['train']:
                res['train'][language_name] = []
            res['train'][language_name].append({
                'question': question,
                'answer': answer,
            })
        for item in aya_dataset['test']:
            question = item['inputs']
            answer = item['targets']
            language = item['language']
            if language == 'Simplified Chinese':
                language = 'Chinese'
            if language not in self.language_map.keys():
                continue

            if language not in res['test']:
                res['test'][language] = []
            res['test'][language].append({
                'question': question,
                'answer': answer,
            })
        return res


class MMMLU_dataset:
    def __init__(self, dataset_path, save_path):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.dataset = self.load_dataset()
        with open(os.path.join(save_path, "MMMLU.json"), "w") as f:
            json.dump(self.dataset, f, indent=4)
    def load_dataset(self):
        res = {}
        with open(self.dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                question = data['question']
                answer = data['answer']
                language = data['language']
                if language not in res:
                    res[language] = []
        return res


class GlobalMMLU_Dataset:
    LANGUAGE_MAP = {
        "am": "Amharic",
        "ar": "Arabic",
        "bn": "Bengali",
        "cs": "Czech",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "fil": "Filipino",
        "fr": "French",
        "ha": "Hausa",
        "he": "Hebrew",
        "hi": "Hindi",
        "ig": "Igbo",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ky": "Kyrgyz",
        "ko": "Korean",
        "lt": "Lithuanian",
        "mg": "Malagasy",
        "ms": "Malay",
        "ne": "Nepali",
        "nl": "Dutch",
        "ny": "Chichewa",
        "fa": "Persian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "si": "Sinhala",
        "sn": "Shona",
        "so": "Somali",
        "es": "Spanish",
        "sr": "Serbian",
        "sw": "Swahili",
        "sv": "Swedish",
        "te": "Telugu",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese",
        "yo": "Yoruba",
        "zh": "Chinese",
    }

    LANGUAGE_MAP_SMALL = {
        "en": "English",
        "zh": "Chinese",
        "ar": "Arabic",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "hi": "Hindi",
        "vi": "Vietnamese",
    }

    ANSWER_KEY_MAP = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}

    def __init__(self, save_path, languages=None, max_samples=None):
        self.save_path = save_path
        self.languages = languages or list(self.LANGUAGE_MAP.keys())
        self.max_samples = max_samples
        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "GlobalMMLU.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        train_dataset = self.dataset['test']
        test_dataset = self.dataset['dev']


        with open(os.path.join(save_path, "GlobalMMLU_train.json"), "w") as f:
            json.dump(train_dataset, f, indent=4, ensure_ascii=False)
        with open(os.path.join(save_path, "GlobalMMLU_test.json"), "w") as f:
            json.dump(test_dataset, f, indent=4, ensure_ascii=False)

    def load_dataset(self):
        from datasets import load_dataset

        res = {'test': {}, 'dev': {}}
        for lang_code in self.languages:
            lang_name = self.LANGUAGE_MAP[lang_code]
            print(f"Loading Global-MMLU for {lang_name} ({lang_code})...")
            ds = load_dataset("CohereLabs/Global-MMLU", lang_code, split="test")

            res['test'][lang_name] = []
            for item in ds:
                if self.max_samples and len(res['test'][lang_name]) >= self.max_samples:
                    break
                answer_letter = item["answer"]
                answer_field = self.ANSWER_KEY_MAP.get(answer_letter)
                answer_text = item[answer_field] if answer_field else answer_letter

                res['test'][lang_name].append({
                    "sample_id": item["sample_id"],
                    "subject": item["subject"],
                    "subject_category": item["subject_category"],
                    "question": item["question"],
                    "option_a": item["option_a"],
                    "option_b": item["option_b"],
                    "option_c": item["option_c"],
                    "option_d": item["option_d"],
                    "answer": answer_letter,
                    "answer_text": answer_text,
                })
            
            ds = load_dataset("CohereLabs/Global-MMLU", lang_code, split="dev")
            res['dev'][lang_name] = []
            for item in ds:
                if self.max_samples and len(res['dev'][lang_name]) >= self.max_samples:
                    break
                answer_letter = item["answer"]
                answer_field = self.ANSWER_KEY_MAP.get(answer_letter)
                answer_text = item[answer_field] if answer_field else answer_letter
                res['dev'][lang_name].append({
                    "sample_id": item["sample_id"],
                    "subject": item["subject"],
                    "subject_category": item["subject_category"],
                    "question": item["question"],
                    "option_a": item["option_a"],
                    "option_b": item["option_b"],
                    "option_c": item["option_c"],
                    "option_d": item["option_d"],
                    "answer": answer_letter,
                    "answer_text": answer_text,
                })
        return res




class MCSQA_Dataset:
    """Multilingual CommonsenseQA (mCSQA) — 5-option MC across 8 languages.
    HuggingFace: yusuke1997/mCSQA
    """

    LANGUAGE_MAP = {
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "de": "German",
        "pt": "Portuguese",
        "nl": "Dutch",
        "fr": "French",
        "ru": "Russian",
    }

    LABEL_TO_FIELD = {"a": "option_a", "b": "option_b", "c": "option_c",
                      "d": "option_d", "e": "option_e"}

    def __init__(self, save_path, languages=None, max_samples=None):
        self.save_path = save_path
        self.languages = languages or list(self.LANGUAGE_MAP.keys())
        self.max_samples = max_samples
        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "mCSQA.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        train_dataset = self.dataset["train"]
        test_dataset = self.dataset["test"]

        with open(os.path.join(save_path, "mCSQA_train.json"), "w") as f:
            json.dump(train_dataset, f, indent=4, ensure_ascii=False)
        with open(os.path.join(save_path, "mCSQA_test.json"), "w") as f:
            json.dump(test_dataset, f, indent=4, ensure_ascii=False)

    def _parse_choices(self, choices):
        """Convert HF choices dict {label: [...], text: [...]} to flat option fields."""
        options = {}
        for label, text in zip(choices["label"], choices["text"]):
            field = self.LABEL_TO_FIELD.get(label.lower())
            if field:
                options[field] = text
        for field in self.LABEL_TO_FIELD.values():
            options.setdefault(field, "")
        return options

    def load_dataset(self):
        from datasets import load_dataset

        res = {"train": {}, "test": {}}
        for lang_code in self.languages:
            lang_name = self.LANGUAGE_MAP[lang_code]
            print(f"Loading mCSQA for {lang_name} ({lang_code})...")
            ds = load_dataset("yusuke1997/mCSQA", lang_code)

            for split_src, split_dst in [("train", "train"), ("test", "test")]:
                res[split_dst][lang_name] = []
                for item in ds[split_src]:
                    if self.max_samples and len(res[split_dst][lang_name]) >= self.max_samples:
                        break
                    options = self._parse_choices(item["choices"])
                    answer_key = item["answerKey"].upper()
                    answer_field = self.LABEL_TO_FIELD.get(item["answerKey"].lower())
                    answer_text = options.get(answer_field, answer_key)

                    res[split_dst][lang_name].append({
                        "id": item["id"],
                        "question": item["question"],
                        "question_concept": item.get("question_concept", ""),
                        "option_a": options["option_a"],
                        "option_b": options["option_b"],
                        "option_c": options["option_c"],
                        "option_d": options["option_d"],
                        "option_e": options["option_e"],
                        "answer": answer_key,
                        "answer_text": answer_text,
                        "hard": item.get("hard", False),
                    })
        return res


class Belebele_Dataset:
    """Belebele — 4-option reading comprehension MC across 122 languages.
    HuggingFace: facebook/belebele
    Each question is paired with a short passage from FLORES-200.
    """

    LANGUAGE_MAP = {
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
        "zul_Latn": "Zulu"
    }

    LANGUAGE_MAP_SMALL = {
        "eng_Latn": "English",
        "zho_Hans": "Chinese",
        "arb_Arab": "Arabic",
        "jpn_Jpan": "Japanese",
        "fra_Latn": "French",
        "deu_Latn": "German",
        "spa_Latn": "Spanish",
        "rus_Cyrl": "Russian",
        "hin_Deva": "Hindi",
        "vie_Latn": "Vietnamese",
    }

    ANSWER_NUM_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}

    def __init__(self, save_path, languages=None, max_samples=None):
        self.save_path = save_path
        self.languages = languages or list(self.LANGUAGE_MAP.keys())
        self.max_samples = max_samples
        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "Belebele.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        with open(os.path.join(save_path, "Belebele_test.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

    def load_dataset(self):
        from datasets import load_dataset

        res = {}
        for lang_code in self.languages:
            lang_name = self.LANGUAGE_MAP[lang_code]
            print(f"Loading Belebele for {lang_name} ({lang_code})...")
            try:
                ds = load_dataset("facebook/belebele", lang_code, split="test")
            except Exception as e:
                continue

            res[lang_name] = []
            for item in ds:
                if self.max_samples and len(res[lang_name]) >= self.max_samples:
                    break
                answer_num = int(item["correct_answer_num"])
                answer_letter = self.ANSWER_NUM_TO_LETTER.get(answer_num, "A")
                answer_field = f"mc_answer{answer_num}"
                answer_text = item.get(answer_field, answer_letter)

                passage = item.get("flores_passage", "")
                question_text = item.get("question", "")
                combined_question = f"{passage}\n\n{question_text}" if passage else question_text

                res[lang_name].append({
                    "link": item.get("link", ""),
                    "question_number": item.get("question_number", ""),
                    "question": combined_question,
                    "passage": passage,
                    "option_a": item.get("mc_answer1", ""),
                    "option_b": item.get("mc_answer2", ""),
                    "option_c": item.get("mc_answer3", ""),
                    "option_d": item.get("mc_answer4", ""),
                    "answer": answer_letter,
                    "answer_text": answer_text,
                })
        return res

class MMath_Dataset:
    """MMath — multilingual math reasoning dataset across 10 languages.
    Data files: {dataset_path}/{lang_code}.json
    """

    LANGUAGE_MAP = {
        "ar": "Arabic",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "th": "Thai",
        "vi": "Vietnamese",
        "zh": "Chinese",
    }

    def __init__(self, dataset_path, save_path, languages=None, max_samples=None):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.languages = languages or list(self.LANGUAGE_MAP.keys())
        self.max_samples = max_samples
        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "MMath.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        with open(os.path.join(save_path, "MMath_test.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

    def load_dataset(self):
        res = {}
        for lang_code in self.languages:
            lang_name = self.LANGUAGE_MAP[lang_code]
            print(f"Loading MMath for {lang_name} ({lang_code})...")
            data_path = os.path.join(self.dataset_path, f"{lang_code}.json")
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            res[lang_name] = []
            for item in data:
                if self.max_samples and len(res[lang_name]) >= self.max_samples:
                    break
                res[lang_name].append({
                    "gid": item.get("gid", ""),
                    "question": item["question"],
                    "answer": item["answer"],
                    "data_source": item.get("data_source", ""),
                    "data_source_id": item.get("data_source_id", ""),
                })
        return res


class MGSM_Dataset:
    """MGSM-Rev2 — Multilingual Grade School Math across 11 languages.
    Data files: {dataset_path}/mgsm_{lang_code}.tsv (tab-separated: question\\tanswer)
    """

    LANGUAGE_MAP = {
        "bn": "Bengali",
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "ja": "Japanese",
        "ru": "Russian",
        "sw": "Swahili",
        "te": "Telugu",
        "th": "Thai",
        "zh": "Chinese",
    }

    def __init__(self, dataset_path, save_path, languages=None, max_samples=None):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.languages = languages or list(self.LANGUAGE_MAP.keys())
        self.max_samples = max_samples
        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "MGSM.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

        with open(os.path.join(save_path, "MGSM_test.json"), "w") as f:
            json.dump(self.dataset, f, indent=4, ensure_ascii=False)

    def load_dataset(self):
        res = {}
        for lang_code in self.languages:
            lang_name = self.LANGUAGE_MAP[lang_code]
            print(f"Loading MGSM for {lang_name} ({lang_code})...")
            data_path = os.path.join(self.dataset_path, f"mgsm_{lang_code}.tsv")

            res[lang_name] = []
            with open(data_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if self.max_samples and len(res[lang_name]) >= self.max_samples:
                        break
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    res[lang_name].append({
                        "id": idx,
                        "question": parts[0],
                        "answer": parts[1],
                    })
        return res


if __name__ == "__main__":
    # dataset = PIQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/piqa/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset")  # 全部语言
    # dataset = PIQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/piqa/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset", languages=["Chinese (Simplified)", "English", "Japanese"], max_samples=100)  # 指定语言子集
    # dataset = MKQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/mkqa.jsonl", save_path="/home/dzhang98/code/Multiling-reasoning/dataset")
    # dataset = Aya_Dataset(dataset_path="", save_path="/home/dzhang98/code/Multiling-reasoning/dataset")


    # PIQA — 全量语言
    # dataset = PIQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/piqa/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=100)
    # PIQA — 指定语言子集
    dataset = PIQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/piqa/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=30)

    # GlobalMMLU — 全量语言
    # dataset = GlobalMMLU_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=100)
    # GlobalMMLU — 精简10语言
    # dataset = GlobalMMLU_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset",
    #                              languages=list(GlobalMMLU_Dataset.LANGUAGE_MAP_SMALL.keys()), max_samples=50)

    # mCSQA — 全量数据
    # dataset = MCSQA_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=100)
    # mCSQA — 每语言最多200条
    # dataset = MCSQA_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=50)

    # Belebele — 全量语言
    # dataset = Belebele_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset",max_samples=30)
    # Belebele — 精简10语言
    # dataset = Belebele_Dataset(save_path="/home/dzhang98/code/Multiling-reasoning/dataset",
    #                            languages=list(Belebele_Dataset.LANGUAGE_MAP_SMALL.keys()), max_samples=50)

    # MMath — 全量语言
    dataset = MMath_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/MMATH/mmath", save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=100)
    # MMath — 精简10语言
    # dataset = MMath_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/mmath/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset",
    #                         languages=list(MMath_Dataset.LANGUAGE_MAP_SMALL.keys()), max_samples=50)

    # MGSM — 全量语言
    dataset = MGSM_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/MGSM-Rev2/MGSM-Rev2", save_path="/home/dzhang98/code/Multiling-reasoning/dataset", max_samples=100)
    # MGSM — 精简10语言
    # dataset = MGSM_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/mgsm/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset",
    #                         languages=list(MGSM_Dataset.LANGUAGE_MAP_SMALL.keys()), max_samples=50)