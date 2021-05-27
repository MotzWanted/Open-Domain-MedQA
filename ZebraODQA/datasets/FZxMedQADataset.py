from typing import Dict
from notebooks.starostka_scratchbook import BATCH_SIZE
import pandas as pd
import numpy as np
import torch
import os
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc

from datasets import load_dataset

# Custom Dataset
BERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
BATCH_SIZE = 32
class FZxMedQADataset(Dataset):
    """
    FindZebra and MedQA dataset
    """

    def __init__(self, data_dir:str, split:str):
        # load files from dictionary
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.dataset = load_dataset("json", data_dir=data_dir, split=split)
        return self.dataset
    
    def __init__(self, data_files:list, split:str):
        # load specified dataset files
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.dataset = load_dataset("json", data_files=data_files, field='data', split=split)
        return self.dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index:int):
        # get single sample from dataset (with encodings applied on-demand)
        return dict(
            none=None
        )

train_dset = FZxMedQADataset(data_dir='medqa_dataset.json', split='train')

# tokenize (see. https://huggingface.co/docs/datasets/quicktour.html)
encode_query = lambda query : tokenizer.encode_plus(
        '[Q]',
        query['question'],
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
train_dset = train_dset.map(encode_query, batched=True)

# get pytorch loaders
import torch
train_dset.set_format(type='torch', columns=['query.input_ids', 'query.attention_mask',
'evidence.input_ids', 'evidence.attention_mask', 'anser_choices.input_ids', 'answer_idx'])
dataloader = torch.utils.data.DataLoader(train_dset, batch_size=32)



# HuggingFace dataset to PyTorch
BERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
dataset = load_dataset("json", data_files="medqa_dataset.json", field='data', split='train')
dataset = dataset.map(lambda e: 
    tokenizer.encode_plus(
        e['question'],
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt
    ),
    batched=True)
dataset.set_format(
    type='torch', 
    columns=['query.input_ids', 
    'query.attention_mask',
    'evidence.input_ids',
    'evidence.attention_mask',
    'answer_choices.input_ids',
    'answer_idx']
)
dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE)