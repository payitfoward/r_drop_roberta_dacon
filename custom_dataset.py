import torch
import pandas as pd
from torch.utils.data import Dataset


class Train_Dataset(Dataset):
    label2idx = {'사실형-긍정-현재-확실': 0,
                '사실형-긍정-과거-확실': 1,
                '사실형-긍정-미래-확실': 2,
                '추론형-부정-현재-확실': 3,
                '예측형-긍정-미래-불확실': 4,
                '추론형-긍정-현재-확실': 5,
                '추론형-긍정-과거-확실': 6,
                '추론형-긍정-현재-불확실': 7,
                '대화형-긍정-미래-확실': 8,
                '사실형-미정-현재-확실': 9,
                '사실형-부정-과거-확실': 10,
                '예측형-부정-과거-확실': 11,
                '추론형-긍정-미래-확실': 12,
                '사실형-긍정-미래-불확실': 13,
                '대화형-긍정-현재-확실': 14,
                '사실형-부정-현재-확실': 15,
                '대화형-긍정-과거-확실': 16,
                '사실형-긍정-과거-불확실': 17,
                '사실형-긍정-현재-불확실': 18,
                '대화형-긍정-현재-불확실': 19,
                '예측형-미정-미래-불확실': 20,
                '예측형-긍정-미래-확실': 21,
                '추론형-부정-미래-확실': 22,
                '사실형-미정-미래-확실': 23,
                '추론형-긍정-미래-불확실': 24,
                '대화형-부정-과거-확실': 25,
                '대화형-긍정-미래-불확실': 26,
                '대화형-미정-미래-불확실': 27,
                '추론형-미정-미래-불확실': 28,
                '추론형-부정-미래-불확실': 29,
                '추론형-부정-과거-확실': 30,
                '사실형-미정-미래-불확실': 31,
                '추론형-긍정-과거-불확실': 32,
                '예측형-긍정-현재-확실': 33,
                '사실형-부정-과거-불확실': 34,
                '예측형-긍정-과거-확실': 35,
                '예측형-긍정-과거-불확실': 36,
                '대화형-긍정-과거-불확실': 37,
                '대화형-미정-과거-불확실': 38,
                '사실형-부정-미래-확실': 39,
                '추론형-부정-현재-불확실': 40,
                '사실형-미정-현재-불확실': 41,
                '대화형-미정-현재-불확실': 42,
                '예측형-부정-현재-불확실': 43,
                '대화형-부정-현재-불확실': 44,
                '예측형-긍정-현재-불확실': 45,
                '추론형-미정-미래-확실': 46,
                '사실형-부정-미래-불확실': 47,
                '추론형-미정-현재-불확실': 48,
                '대화형-부정-현재-확실': 49,
                '사실형-미정-과거-확실': 50,
                '추론형-부정-과거-불확실': 51,
                '사실형-부정-현재-불확실': 52,
                '대화형-부정-미래-확실': 53,
                '예측형-미정-현재-확실': 54,
                '예측형-미정-현재-불확실': 55,
                '예측형-부정-미래-불확실': 56,
                '대화형-미정-미래-확실': 57,
                '대화형-미정-과거-확실': 58,
                '추론형-미정-현재-확실': 59,
                '대화형-부정-과거-불확실': 60,
                '추론형-미정-과거-불확실': 61,
                '예측형-미정-미래-확실': 62,
                '예측형-미정-과거-확실': 63}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in data.label.tolist():
            labels.append(self.label2idx[i])
        return labels
    
class Test_Dataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        features = self.tokenizer(
            data.문장.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for i in range(len(data)):
            labels.append(0)
        return labels