import os

from json import dump
from os.path import dirname
from pathlib import Path
from typing import List

from datasets import load_dataset, load_from_disk

class MedQACorpus(Corpus):
    def __init__(self, tokenizer):
        # try:
        #     self.corpus = load_from_disk(os.path.join(Config.cache_dir, "wikipedia/"))
        #     print("\nReusing previously saved the dataset")
        # except FileNotFoundError:
        print("\nDownloading the dataset")
        super().__init__(tokenizer)
        corpus = load_dataset(
            'json', 
            data_files=os.path.join(gdrive_data_medqa,"medqa_corpus.json"), 
            field='data',
            block_size=40<<20, 
            split='train[:18]',
            cache_dir=config_cache_dir)

        corpus = corpus.map(
            lambda data: {
                "title": data['title'],
                "context": data['text']
            },
            remove_columns=corpus.column_names,
            num_proc=config_max_proc_to_use
        )

        self.corpus = corpus.map(
            self.generate_block_info,
            batched=True,
            remove_columns=corpus.column_names,
            num_proc=config_max_proc_to_use
        )

        self.corpus.save_to_disk(os.path.join(config_cache_dir, "medqa/"))

    def get_corpus(self):
        return self.corpus