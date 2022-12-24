import pandas as pd
import numpy as np
import torch
import os

from transformers import AutoTokenizer, AutoConfig, TrainingArguments

from custom_trainer import Custom_Trainer
from custom_dataset import Train_Dataset, Test_Dataset
from custom_model import RobertaForSequenceClassification

from sklearn.model_selection import StratifiedKFold

from utils import set_allseed, Augmenter, compute_metrics

import warnings

class cfg():
    seed = 42
    batch_size = 32
    total_batch_size = 128
    gradient_step = total_batch_size / batch_size
    save_steps = 103
    epochs = 5
    lr = 1e-4
    wd = 1e-4
    save_model_limit = 2
    num_labels = None
    model_init = "klue/roberta-large"

set_allseed(cfg.seed)
warnings.filterwarnings(action='ignore')

train_df = pd.read_csv("train.csv")
train_df = train_df[["문장","label"]]
cfg.num_labels = len(train_df.label.value_counts)

test_df = pd.read_csv("test.csv")

kfold_function = StratifiedKFold(n_splits=5, random_state=cfg.seed, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(cfg.model_init)
config = AutoConfig.from_pretrained(cfg.model_init)
config.num_labels = cfg.num_labels
config.cls_token_id = tokenizer.cls_token_id
config.sep_token_id = tokenizer.sep_token_id

aug = Augmenter(tokenizer = tokenizer)

training_args = TrainingArguments(
    output_dir="./output_model",
    seed = cfg.seed,
    save_total_limit = cfg.save_model_limit,
    save_steps = cfg.save_steps,
    num_train_epochs = cfg.epochs,
    learning_rate = cfg.lr,
    per_device_train_batch_size = cfg.batch_size,
    per_device_eval_batch_size = cfg.batch_size,
    gradient_accumulation_steps = cfg.gradient_step,
    weight_decay = cfg.wd,
    logging_dir="./logs",
    logging_steps = cfg.save_steps,
    evaluation_strategy = "steps",
    eval_steps = cfg.save_steps,
    load_best_model_at_end=True,
)
test_dataset = Test_Dataset(data=test_df, tokenizer=tokenizer)

logit = 0
for i, (train_index, test_index) in enumerate(kfold_function.split(train_df.문장,train_df.label)):
    model = RobertaForSequenceClassification.from_pretrained(cfg.model_init, config=config)
    train_corpus, valid_corpus = train_df.문장[train_index], train_df.문장[test_index]
    train_label, valod_label = train_df.label[train_index], train_df.label[test_index]
    fold_train = pd.concat([train_corpus, train_label], axis =1)
    fold_valid = pd.concat([valid_corpus, valod_label], axis =1)
    augment_data = aug(fold_train).reset_index()
    train_dataset = Train_Dataset(data=augment_data, tokenizer=tokenizer)
    valid_dataset = Train_Dataset(data=fold_valid, tokenizer=tokenizer)

    trainer = Custom_Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    )

    trainer.train()

    logit += trainer.predict(test_dataset).predictions / 5