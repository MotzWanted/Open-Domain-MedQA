import json, os, re, requests
from utils.es_functions import *
from tqdm import tqdm
import gdown

def getDocChunks(article, chunkSize=100, stride=80):
    article = article.replace('\n', ' ').replace('\r', '')
    doc = article.split()

    i = 0
    docChunks = []
    while i < len(doc):
        j = i + chunkSize
        tokens = doc[i : j]
        docChunks.append(' '.join(tokens))
        i += stride
    return docChunks

def is_positive(document, answer, synonyms):
    if re.search(rf"\b{answer}\b", document, re.IGNORECASE):
        return True 
    elif any(re.search(rf"\b{name}\b", document, re.IGNORECASE) for name in synonyms):
        return True

    return False

def ingest_all(data, dateset_name):
    es_create_index(dateset_name)
    for key in tqdm(data.keys()):
        if data[key]['FZ_results']:
            for article in data[key]['FZ_results']:
                docs = getDocChunks(article['doc_context'], chunkSize=100, stride=50)
                for doc in docs:
                    _ = es_ingest(dateset_name, article['title'], doc)
            
    
train_url = 'https://drive.google.com/uc?id=1K8Lu0rI2rK-WZFLxmuQ60mRNWIbSmiy2'
dev_url = 'https://drive.google.com/uc?id=16sJUgYCVwYSp5Zy35xW7NlUUBGhDNdWO'
test_url = 'https://drive.google.com/uc?id=1WZFwLpM_2RNHP2QE-JHlCm5mcb7I0FtN'

output = ['data/dataset/train.json', 'data/dataset/dev.json', 'data/dataset/test.json']
gdown.download(train_url, output[0], quiet=False)
gdown.download(dev_url, output[1], quiet=False)
gdown.download(test_url, output[2], quiet=False)

with open('data/dataset/train.json',"rb") as f:
    train = json.load(f)

with open('data/dataset/dev.json',"rb") as f:
    val = json.load(f)

with open('data/dataset/test.json',"rb") as f:
    test = json.load(f)

datasets = [train, val, test]
ds_names = ["train", "val","test"]

counter = 0
for ds_id, ds in enumerate(datasets):
    out = {
        'version': '0.0.1',
        'data': []
        }
    
    print("Ingesting all articles to ElasticSearch")
    ingest_all(ds, ds_names[ds_id])
    print("Finish ingesting all articles")

    for key in tqdm(ds.keys()):
        if ds[key]['FZ_results']:
            counter+=1
            q_id = key[1:]
            is_golden = False
            synonyms = set()
            length_ = 0
            answer_options = [ds[key]['answer_options'][opt] for opt in ds[key]['answer_options'].keys()]
            
            es_res = es_search(ds_names[ds_id], ds[key]['question'], 100)
            
            for hit in es_res['hits']:
                if is_positive(hit['_source']['text'], ds[key]['answer'], synonyms) and is_golden==False:
                    out['data'].append({
                        'idx' : counter,
                        'question_id' : q_id,
                        'question' : ds[key]['question'],
                        'answer_choices' : answer_options,
                        'answer_idx' : answer_options.index(ds[key]['answer']),
                        'document' : hit['_source']['title'] + ' ' + hit['_source']['text'],
                        'is_gold' : True
                    })
                    is_golden=True

                out['data'].append({
                        'idx' : counter,
                        'question_id' : q_id,
                        'question' : ds[key]['question'],
                        'answer_choices' : answer_options,
                        'answer_idx' : answer_options.index(ds[key]['answer']),
                        'document' : hit['_source']['title'] + ' ' + hit['_source']['text'],
                        'is_gold' : False
                    })

    es_remove_index(ds_names[ds_id])

    with open(ds_names[ds_id]+'_FZ-MedQA.json',"w") as file:
        json.dump(out, file, indent=6)