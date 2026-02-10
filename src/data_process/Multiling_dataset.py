import json
import os
import pandas as pd

class PIQA_Dataset:
    def __init__(self, dataset_path, save_path):
        self.language_map = {
            "Chinese": "cmn_hans.tsv",
            "Arabic": "arb_arab.tsv",
            "Japanese": "jpn_jpan.tsv",
            "Vietnamese": "vie_latn.tsv",
            "Marathi": "mar_deva.tsv",
            "Amharic": "amh_ethi.tsv",
            "Telugu": "tel_telu.tsv",
        }
        self.dataset_path = dataset_path



        self.dataset = self.load_dataset()

        with open(os.path.join(save_path, "PIQA.json"), "w") as f:
            json.dump(self.dataset, f, indent=4)


    def load_dataset(self):
        result = {}
        for language,file_name in self.language_map.items():
            result[language] = []
            data_path = os.path.join(self.dataset_path,file_name)
            df = pd.read_csv(data_path,sep="\t")

            for row in df.itertuples(index=True):
                # question = f"{row.prompt} \n A. {row.solution0} \n B. {row.solution1}"
                question = f"{row.prompt}"

                # result.append({
                #     "question": question,
                #     "opention_0": row.solution0,
                #     "opention_1": row.solution1,
                #     "answer": row.label,
                #     "language": language,
                # })
                option_0 = row.solution0
                option_1 = row.solution1
                answer = row.label

                result[language].append({
                    "question": question,
                    "answer": option_0 if answer == 0 else option_1,

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


if __name__ == "__main__":
    dataset = PIQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/piqa/data", save_path="/home/dzhang98/code/Multiling-reasoning/dataset")
    dataset = MKQA_Dataset(dataset_path="/home/dzhang98/code/Multiling-data/mkqa.jsonl", save_path="/home/dzhang98/code/Multiling-reasoning/dataset")