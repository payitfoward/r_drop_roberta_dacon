from tqdm import tqdm
import collections
import pandas as pd
import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score


def set_allseed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Augmenter():

    def __init__(self, 
                tokenizer, 
                max_num: int = 3000,
                min_num: int = 20,
                reduction: float = 0.8,
                undersampling_ratio: float = 0.8
                ):
        self.tokenizer = tokenizer
        self.max_num = max_num
        self.min_num = min_num
        self.reduction = reduction
        self.undersampling_ratio = undersampling_ratio
        self.punct = [".", ";", "?", ":", "!", ","]

    def __call__(self, dataset):

        labels = dataset.label
        label_ids = collections.defaultdict(list)
        for i, l in enumerate(labels):
            label_ids[l].append(i)

        total_id_list = []
        for l in label_ids:
            id_list = label_ids[l]
            previous_size = len(id_list)  # 기존 데이터 갯수

            if previous_size > self.min_num:
                sample_size = int(len(id_list) * self.reduction)
                augmentated_id_list = random.sample(id_list, sample_size)

                if len(augmentated_id_list) > self.max_num:
                    augmentated_id_list = random.sample(
                        augmentated_id_list,
                        int(len(augmentated_id_list) * self.undersampling_ratio)
                    )

                # status 0 : aug x, status : 1 aug o
                augmentated_id_list = [(a_id, 0)
                                       for a_id in augmentated_id_list]

            else:
                augmentated_id_list = [(a_id, 0) for a_id in id_list]
                while len(id_list) < self.min_num:
                    id_list = id_list * 2

                augmentated_id_list = augmentated_id_list + \
                    [(a_id, 1) for a_id in random.sample(
                        id_list, self.min_num - previous_size)]

            total_id_list.extend(augmentated_id_list)

        org_size, add_size = 0, 0
        for _, status in total_id_list:
            if status == 0:
                org_size += 1
            else:
                add_size += 1

        random.shuffle(total_id_list)
        total_dataset = []
        for i, status in tqdm(total_id_list):
            data = self.augment(dataset.loc[i], status)
            total_dataset.append(data)

        df = pd.DataFrame(total_dataset)
        return df

    # aeda >> koeda library
    def aeda(self, data):
        sentence = data.문장
        insert_size = np.random.randint(1, len(sentence) // 3)

        chars = list(sentence)
        while insert_size > 0:
            punct_id = np.random.randint(len(self.punct))
            punct = self.punct[punct_id]
            insert_id = np.random.randint(len(chars))
            chars = chars[:insert_id] + [punct] + chars[insert_id:]

            insert_size -= 1

        sentence = ''.join(chars)
        data.문장 = sentence
        return data

    # 문장 내의 몇몇 단어들을 임의의 단어로 변환
    def change(self, data):
        sentence = data.문장
        tokens = self.tokenizer.encode(sentence)[1:-1]
        change_size = int(len(tokens) * 0.15)

        if change_size > 0:
            change_ids = random.sample(range(len(tokens)), change_size)
            for c_id in change_ids:
                tokens[c_id] = np.random.randint(len(self.tokenizer))

            data.문장 = self.tokenizer.decode(tokens)

        return data

    # 문장의 순서 바꾸기
    def reverse(self, data):
        sentence = data['문장']
        words = sentence.split(' ')

        if len(words) > 5:
            index = np.random.randint(1, len(words) - 1)
            reversed = words[index:] + words[:index]
            sentence = ' '.join(reversed)

        data['문장'] = sentence
        return data

    # 문장의 단어 삭제
    def delete(self, data, del_ratio=0.2):

        sentence = data.문장
        words = sentence.split(' ')

        if len(words) > 5:
            word_size = len(words)
            del_size = int(word_size * del_ratio)
            del_indices = random.sample(range(word_size), del_size)

            deleted = []
            for i, word in enumerate(words):
                if i in del_indices:
                    continue
                deleted.append(word)

            sentence = ' '.join(deleted)

        data.문장 = sentence
        return data

    def augment(self, data, status):
        if status == 0:
            return data
        else:
            option = np.random.randint(4)
            if option == 0:
                data = self.aeda(data)
            elif option == 1:
                data = self.reverse(data)
            elif option == 2:
                data = self.delete(data)
            else:
                data = self.change(data)

            return data

def compute_metrics(pred, num_labels):
    predict = pred.predictions.argmax(axis=1)
    ref = pred.label_ids
    pred_li, ref_li = [], []
    for i, j in zip (predict, ref):
        prediction, reference = [0] * num_labels, [0] * num_labels
        prediction[i] = 1
        reference[j] = 1
        pred_li.append(prediction)
        ref_li.append(reference)
    f1 = f1_score(pred_li, ref_li, average="weighted")
    return {'f1' : f1 }