
import requests
from bs4 import BeautifulSoup
import time
import pickle
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from SparseMethods import SparseMethods

import os
dirname = os.getcwd()
data_path = os.path.normpath(join(os.getcwd(), "./data_clean/dicts/"))
text_path = os.path.normpath(
    join(os.getcwd(), "./data_clean/textbooks/en_sentence_100/"))

index_name = 'docs_bm25_shards5_new' # 100 tokens
base_url = "http://localhost:9200/"

def search_docs(query_input, n, index_name):
    url = base_url + index_name + "/_doc/_search"

    x = {
        "query": {
            "match": {
                "content": query_input
            }
        },
        "from": 0,
        "size": n
    }

    payload = json.dumps(x)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    out = json.loads(response.text)
    number_of_hits = len(out['hits']['hits'])
    results = []
    for i in range(number_of_hits):
        score = out['hits']['hits'][i]['_score']
        paragraph = out['hits']['hits'][i]['_source']
        result = {
            "score": score,
            "evidence": paragraph
        }
        results.append(result)

    return results

def retrieve_positive_passage(question, answer_option):
    # retrieving from elasticsearch
    query = question + " " + answer_option
    result = search_docs(query, N, index_name)
    return {'question_ai': query, 'pos_passage': " ".join(pos['evidence']['content'] for pos in result)}

def construct_dataset(dictionary):
    construct = dict()
    idx = 0
    for q_id in dictionary.keys():
        for _, answer in dictionary[q_id]['opt'].items():
            # construct datastructure for bert
            positive_passages = retrieve_positive_passage(dictionary[q_id]['q'], answer)
            # mutate the new dictionary (use answer option as key to related positive passages)
            construct[idx] = positive_passages
            idx += 1
    # dump to file
    pickle.dump(construct, open("positive_passages.pickle", "wb"))

f = open(data_path + "/dev.pickle", "r")
dev_dict = pickle.load(f)
N = 5
retrieve_positive_passage("Cells", "CT-Scan")


