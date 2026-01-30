import json

class MathMistakeDataset:
    def __init__(self, data_path: str):
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]