import json
import os
import requests
from bs4 import BeautifulSoup
import time
import pickle
import numpy as np 
import matplotlib.pyplot as plt

cwd = os.getcwd()
data_folder = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/questions/US/metamap_extracted_phrases/test/" 
key = "57796ff4-22c1-4b43-93ce-2fb76600510c"

with open(data_folder+"phrases_test.jsonl", 'r') as json_file:
    json_list = list(json_file)
results = json.loads(json_list[0])
#scores = []
#sources = []
#dsc_dict = {}
#train_dict = {} #2.7 timer
#dev_dict = {} #0.3 timer
test_dict = {} #0.67
start_time = time.time()
for i, value in enumerate(json_list):
    idx = 'q' + str(i+10178+1272)#10178+1272
    result = json.loads(value)
    
    test_dict[idx] = {}
    test_dict[idx]['qst'] = result['question']
    test_dict[idx]['ans'] = result['answer']
    test_dict[idx]['opt'] = result['options']
    test_dict[idx]['metamap'] = result['metamap_phrases']

    #query = {'api_key' : key, 'q' : result['answer']}
    #try:
        #response = requests.get("https://www.findzebra.com/api/v1/query",query)
    
        #content = response.json()
        #doc = BeautifulSoup(content['response']['docs'][0]['display_content'])

        #test_dict[idx] = {}
        #test_dict[idx]['qst'] = result['question']
        #test_dict[idx]['ans'] = result['answer']
        #test_dict[idx]['opt'] = result['options']
        #test_dict[idx]['doc'] = doc.get_text()
        #test_dict[idx]['sco'] = content['response']['docs'][0]['score']
        #test_dict[idx]['src'] = content['response']['docs'][0]['source']

        #scores.append(content['response']['docs'][0]['score'])
        #sources.append(content['response']['docs'][0]['source'])
    
    #except:
        #dsc_dict[idx] = {}

        #dsc_dict[idx]['qst'] = result['question']
        #dsc_dict[idx]['ans'] = result['answer']
        #dsc_dict[idx]['opt'] = result['options']
end_time = (time.time() - start_time)

data_path = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/dicts/"

with open(data_path + 'test.pickle',"wb") as file:
    pickle.dump(test_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(data_path + '/train.pickle',"rb") as file:
    train = pickle.load(file)

with open(data_path + '/dev.pickle',"rb") as file:
    dev = pickle.load(file)

with open(data_path + '/test.pickle',"rb") as file:
    test = pickle.load(file)

scores = []
for d in [train, dev, test]:
    for k in d.keys():
        scores.append(d[k]['sco'])

(len(train)+len(dev)+len(test))/(len(train)+len(dev)+len(test)+len(dsc))

plt.hist(scores, density=False, bins=int(np.sqrt(len(scores))))  # density=False would make counts
plt.ylabel('Frequency')
plt.xlabel('Scores')
plt.show() 

np.mean(scores)

list(train.keys())[:60]

key = "57796ff4-22c1-4b43-93ce-2fb76600510c"
query = {'api_key' : key, 'q' : train['q20']['ans']}

response = requests.get("https://www.findzebra.com/api/v1/query",query)
content = response.json()
title = content['response']['docs'][4]['title']
doc = BeautifulSoup(content['response']['docs'][4]['display_content'])

scores = []
for i in range(0,10):
    scores.append(content['response']['docs'][i]['score'])

train['q20']['qst']
train['q20']['ans']
np.mean(scores)
np.min(scores)
np.max(scores)
