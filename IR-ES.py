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
    join(os.getcwd(), "./data_clean/textbooks/en_sentence/"))

index_name = 'docs_bm25'

################ create index ################
base_url = "http://localhost:9200/"

body = {
    "settings": {
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25"
                }
            }
        }
    }
}
payload = json.dumps(body)
headers = {
    'Content-Type': 'application/json'
}

response = requests.request(
    "PUT", base_url+index_name, headers=headers, data=payload)

################ ingest docs ################

textbooks = [f for f in listdir(text_path) if isfile(join(text_path, f))]
with open(text_path + "/" + textbooks[0], "rb") as file:
    textbook = pickle.load(file)
# sentence_list = []
# for book in textbooks:
#     with open(data_path + book,"rb") as file:
#         textbook = pickle.load(file)
#     sentence_list.append(sent for sent in textbook)


def ingest_docs(sentence_list, index_name):
    counter = 1
    for sent in tqdm(sentence_list):
        x = {
            "content": sent
        }
        payload = json.dumps(x)
        headers = {
            'Content-Type': 'application/json'
        }

        url = base_url + index_name + "/_doc/" + str(counter)

        response = requests.request("PUT", url, headers=headers, data=payload)

        counter += 1


ingest_docs(textbook, index_name)

################ search docs ################


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

################ get accuracies ################


with open(data_path + '/train.pickle', "rb") as file:
    train_dict = pickle.load(file)

with open(data_path + '/dev.pickle', "rb") as file:
    dev_dict = pickle.load(file)

with open(data_path + '/test.pickle', "rb") as file:
    test_dict = pickle.load(file)

# number of results to return
n = 5

def get_accuracy(data_dict, index_name):
    y_hat = []
    for key in tqdm(list(data_dict.keys())):
        q = data_dict[key]['qst']
        ans_opt = [data_dict[key]['opt'][i]
                   for i in data_dict[key]['opt'].keys()]

        scores = []
        for a in ans_opt:
            query_input = q + " " + a
            out = search_docs(query_input, n, index_name)
            scores.append(sum(e['score'] for e in out))

        if ans_opt[np.argmax(scores)] == data_dict[key]['ans']:
            print(ans_opt[np.argmax(scores)], " = ", data_dict[key]['ans'])
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    return np.mean(y_hat)


get_accuracy(dev_dict, index_name)

################ CUSTOM get accuracies ################

with open(data_path + '/questions/US/metamap_extracted_phrases/dev/phrases_dev', "r") as file:
    meta_dev_dict = pickle.load(file)

def get_reweighted_accuracy(data_dict, index_name, ngram_func):
    y_hat = []
    for key in tqdm(list(data_dict.keys())):
        q = data_dict[key]['qst']
        ans_opt = [data_dict[key]['opt'][i]
                   for i in data_dict[key]['opt'].keys()]

        # compute average query length
        q_lengths = []
        for a in ans_opt:
            q_i = q + " " + a
            q_lengths.append(len(SparseMethods.prep_query(q_i)))
        avgQueryLen = np.mean(q_lengths)

        scores = []
        for a in ans_opt:
            query_input = q + " " + a
            out = search_docs(query_input, n, index_name)

            # prepare for re-weight
            Passages = [p['evidence']['content'] for p in out]
            DFD = []
            d_lengths = []
            for p in Passages:
                D = SparseMethods.prep_document(p)
                # precompute n-grams document frequency distributions for passages/documents
                DFD.append(SparseMethods.ngram_freq_dist(D, ngram_func))
                d_lengths.append(len(D))
            avgDocLen = np.mean(d_lengths)
            # create sparseMethods instance
            sparseMethods = SparseMethods(
                DFD, avgQueryLen, avgDocLen, ngram_func)

            # results for each passage
            Q = SparseMethods.prep_query(query_input)
            S_scores = []
            for item in out:
                D = SparseMethods.prep_document(item['evidence']['content'])
                S_scores.append(sparseMethods.S(Q, D) + item['score'])

            scores.append(sum(s for s in S_scores))

        if ans_opt[np.argmax(scores)] == data_dict[key]['ans']:
            print(ans_opt[np.argmax(scores)], " = ", data_dict[key]['ans'])
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    return np.mean(y_hat)

get_reweighted_accuracy(dev_dict, index_name, SparseMethods.gen_unigrams)
