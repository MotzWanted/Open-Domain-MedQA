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

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

# check current gpu
os.system("nvidia-smi")

# inspect data
with open("medqa_dataset.json") as df_file:
    df = json.load(df_file)
df = pd.DataFrame.from_dict(df['data'])
df.head()

# HuggingFace dataset to PyTorch
from datasets import load_dataset
BERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
dataset = load_dataset("json", data_files="medqa_dataset.json", field='data', split='train')
dataset = dataset.map(lambda e: tokenizer(), batched=True)

# splitting
train_df, val_df = train_test_split(df, test_size=0.05)
train_df.shape, val_df.shape

# create sample
train_df_sample = train_df.sample(365)
sample_row = df.iloc[10]
sample_question = sample_row.question
sample_question

# load bert
BERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1" # https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# encoder definition
encoding = tokenizer.encode_plus(
    sample_question,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding="max_length",
    return_attention_mask=True,
    return_tensors='pt'
)
encoding.keys()
encoding.input_ids.shape, encoding.attention_mask.shape

# inspect ids and attension mask of 20 first tokens
encoding.input_ids.squeeze()[:20]
encoding.attention_mask.squeeze()[:20]

# convert ids to actual tokens
tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze())[:30]

# create pytorch dataset
class MedZebraDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
    
    def __len__(self):
        # size of dataset
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        symptoms_text = data_row.symptoms
        question_text = data_row.question

        encoding = self.tokenizer.encode_plus(
            sample_question,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            question_text=question_text,
            input_ids=encoding.input_ids.flatten(), # flatten the dimensions
            attention_mask=encoding.attention_mask.flatten()
        )

# instantiate the dataset
train_dataset = MedZebraDataset(train_df, tokenizer)
sample_item = train_dataset[0]
sample_item.keys()

# test basic prediction
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
sample_item['input_ids'].unsqueeze(dim=0).shape # expand the dimension to imitate a batch
prediction = bert_model(sample_item['input_ids'].unsqueeze(dim=0), sample_item['attention_mask'].unsqueeze(dim=0))
prediction.last_hidden_state.shape, prediction.pooler_output.shape

# create lightning datamodule
class MedZebraDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
    
    def setup(self):
        self.train_dataset = MedZebraDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = MedZebraDataset(
            self.test_df,
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
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

# instantiate the datamodule
N_EPOCHS = 10
BATCH_SIZE = 32

data_module = MedZebraDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

# create model module
class MedZebraODQA(pl.LightningModule):
    def __init__(self, steps_per_epoch=None, n_epochs=None) -> None:
        super().__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        #self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        output  = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss, outputs = self(input_ids, attention_mask) # calls the forward method
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs}
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss, outputs = self(input_ids, attention_mask) # calls the forward method
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs}
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss, outputs = self(input_ids, attention_mask) # calls the forward method
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps
        )
        return [optimizer], [scheduler]

# create instance of model
model = MedZebraODQA(
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    n_epochs=N_EPOCHS
)

_, predictions = model(
    sample_item["input_ids"].unsqueeze(dim=0),
    sample_item["attention_mask"].unsqueeze(dim=0)
)
predictions

# train the model
trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, progress_bar_refresh_rate=30)
trainer.fit(model, data_module)