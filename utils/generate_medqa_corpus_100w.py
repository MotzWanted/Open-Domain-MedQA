from dataclasses import field
import sys
import ijson
import json
from tqdm import tqdm
from os import getenv, path, getcwd
import re

c_file = open("./corpus/medqa_corpus.json", 'r')
objects = ijson.items(c_file, 'data.item')
books = list(objects)

# split content of each book into 100words and write it to file
corpus = {
    'version': '0.0.1',
    'data': []
}

for book in tqdm(books):
    title = book['title']
    text = re.sub(r'\n+',' ', book['text'])
    text = text.split(' ')

    i:int = 0
    sentence = []
    while i < len(text):            
        j = i + 100
        temp_list = text[i:j]
        string = ' '.join(temp_list)
        structure = {'title': title, 'sentence': string}
        corpus['data'].append(structure)
        i += 50

# writing to file
corpus_file = open("medqa_corpus_100w.json", 'w')
json.dump(corpus, corpus_file)
corpus_file.close()
c_file.close()


from datasets import load_dataset
from transformers import BertModel, PreTrainedTokenizerFast

corpus = load_dataset('json',data_files=path.realpath('.')+"/medqa_corpus_construction/medqa_100w_corpus.json", field='data')