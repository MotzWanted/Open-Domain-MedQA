import json, os, re, request
import .es_functions
from tqdm import tqdm

def getDocChunks(article, title, chunkSize=100, stride=50):
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

dev_url = 'https://raw.githubusercontent.com/MotzWanted/Open-Domain-MedQA/master/data/dataset/dev.json'
dev_resp = requests.get(dev_url)
dev = json.loads(dev_resp.text)

datasets = [dev]
ds_names = ["dev"]

counter = 0

for ds_id, ds in enumerate(datasets):
    out = {
    'version': '0.0.1',
    'data': []
    }
    for key in tqdm(list(ds.keys())[:100]):
        q_id = int(key[1:]) 
        es_create_index(q_id)

        if ds[key]['FZ_results']:
            is_golden = False
            synonyms = set()
            length_ = 0

            for article in ds[key]['FZ_results']:
                synonyms.add(ds[key]['FZ_results']['synonyms'])
                docs = getDocChunks(article, ds[key]['FZ_results']['title'], chunkSize=100, stride=50)
                
                length_ += len(docs)
                for doc in docs:
                    es_ingest(q_id, doc)
                
                
            es_res = es_search(q_id, dev[key]['question'], length_)
            
            for hit in es_res['hits']:
                counter += 1
                if is_golden==False:
                    is_golden = is_positive(hit['text'], dev[key]['answer'], synonyms)
                    
                    out['data'] = {
                        'idx' : counter
                        'question_id' : q_id,
                        'question' : dev[key]['question'],
                        'answer_choices' : dev[key]['answer_options'],
                        'answer_idx' : dev[key]['answer_options'].index(dev[key]['answer']),
                        'document' : hit['title'] + ' ' + hit['text'],
                        'is_gold' : is_golden
                    }

                else:
                    out['data'] = {
                            'idx' : counter
                            'question_id' : q_id,
                            'question' : dev[key]['question'],
                            'answer_choices' : dev[key]['answer_options'],
                            'answer_idx' : dev[key]['answer_options'].index(dev[key]['answer']),
                            'document' : hit['title'] + ' ' + hit['text'],
                            'is_gold' : False
                        }

            es_remove_index(q_id)

    with open(ds_names[ds_id]+'_FZ-MedQA.json',"w") as file:
        json.dump(out, file, indent=6)