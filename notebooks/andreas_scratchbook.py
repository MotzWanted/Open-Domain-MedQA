import pandas as pd
import numpy as np
import torch
import os
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc

data_folder = os.getcwd()+"/data/dataset/"

with open(data_folder+'train.json',"rb") as f:
    train = json.load(f)

with open(data_folder+'dev.json',"rb") as f:
    val = json.load(f)

with open(data_folder+'test.json',"rb") as f:
    test = json.load(f)

# load bert model and initiate a tokenizer
BERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# create pytorch dataset
class MedZebraDataset(Dataset):
    def __init__(self, data, tokenizer: BertTokenizer, max_token_len: int = 200):
        self.data = list(data.values())
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
    
    def __len__(self):
        # size of dataset
        return len(self.data)
    
    def __getitem__(self, index: int):
        qst = self.data[index]

        query_input = ['[Q]'+ ' ' + qst['question'] + ' ' + qst['answer_options'][opt_idx] for opt_idx in range(4)]

        encoding = self.tokenizer.encode_plus(
            query_input,    # Sentence to encode.
            add_special_tokens = True,     # Add '[CLS]' and '[SEP]'
            max_length = self.max_token_len,
            padding='max_length',
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',          # Return pytorch tensors.
        )

        return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in encoding.items()}

# instantiate the dataset
validation_dataset = MedZebraDataset(val, tokenizer, 200)
sample_item = validation_dataset[0]
validation_dataset.keys()

# create lightning datamodule
class MedZebraDataModule(pl.LightningDataModule):
    def __init__(self, train_json, val_json, test_json, tokenizer, batch_size=8, max_token_len=200):
        super().__init__()
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
    
    def setup(self):
        self.train_dataset = MedZebraDataset(
            self.train_json,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = MedZebraDataset(
            self.val_json,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = MedZebraDataset(
            self.test_json,
            self.tokenizer,
            self.max_token_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

# instantiate the dataset
validation_dataset = MedZebraDataset(val, tokenizer, 200)
sample_item = validation_dataset[0]
validation_dataset.keys()

ss1 = list(val.values())
ss1[0]