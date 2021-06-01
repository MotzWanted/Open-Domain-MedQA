import pandas as pd
import gdown
import os
from elasticsearch import Elasticsearch
from tqdm import tqdm
import spacy
import scispacy

CACHE = ".cache/"
CHUNK_SIZE = 100
STRIDE=50
es = Elasticsearch()

# Disease related TUIs
disease_tuis = ["T020", "T190", "T049", "T019", "T047", "T050", "T033", "T037", "T048", "T191", "T046", "T184"]

def getDocChunks(article, chunkSize=CHUNK_SIZE, stride=STRIDE):
    article = article.replace("\n", " ").replace("\r", "")
    doc = article.split()

    i = 0
    docChunks = []
    while i < len(doc):
        j = i + chunkSize
        tokens = doc[i:j]
        docChunks.append(" ".join(tokens))
        i += stride
    return docChunks

# Ingest MedQA and FindZebra to ElasticSearch

# Download Corpus data
#gdown.download(dev_url, "data/" + "medqa_dataset.json", quiet=False)
corpus_url = "https://drive.google.com/uc?id=1VX8ew7IRQPbSIb-bbFfhpJ5qwYYqznt_"
gdown.cached_download(corpus_url, os.path.join(CACHE, "findzebra_corpus.json"), quiet=False)
df = pd.read_json(os.path.join(CACHE, "findzebra_corpus.json"))
print(df.iloc[0].raw_content)

es_index = "findzebra"
es.indices.create(index=es_index)
nlp = spacy.load('en_core_sci_scibert')
for idx, sample in tqdm(df.iterrows()):
    sentences = " ".join(nlp(sample['raw_content']).sents)
    chunks = getDocChunks(sentences)
    for chunk in chunks:
        doc = {
        'title': sample['title'],
        'text': chunk,
        'cui': sample['cui']
        }
        response = es.index(index=es_index, body=doc)

corpus_url = "https://drive.google.com/uc?id=1ULBr-CgKuvF-CfQo4F_uKnoPPgbb8YA4"
gdown.cached_download(corpus_url, os.path.join(CACHE, "textbooks.zip"), postprocess=gdown.extractall, quiet=False)
df = pd.read_json(os.path.join(CACHE, "medqa_corpus.jsonl"), lines=True)
df.head()

# Run through each question in given MedQA dataset (dev, test, train)

# Fetch