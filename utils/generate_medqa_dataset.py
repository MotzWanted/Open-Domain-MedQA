import sys
import ijson
import json
from tqdm import tqdm
from os import getenv, path
import re

# Map the MedQA dataset to features of Squad

# open file descriptors for 
c_file = open("medqa_dataset_construction/questions/US/US_qbank.jsonl", 'r')

dataset = {
    'version': '0.0.1',
    'data': []
}

d_file = open("medqa_dataset", "w")

id = 0
for line in tqdm(c_file):
    sample:json = json.loads(line)

    sample_question = re.sub(r'\n+', ' ', sample['question'])
    question = sample_question[sample_question.rfind('.')+2:sample_question.find('?')+1]
    symptoms = sample_question.replace(question, '')
    answers = [sample['options'].get(sample['answer'])] # we may want to add more answers ex. using MetaMap

    for key, option in sample['options'].items():
        structure = {
            'id': id,
            #'title': # we don't have title of article or section where the answer was given...
            'symptoms': symptoms, # patient symptoms
            'question': question + ' ' + option,
            'answers': answers
        }
        dataset['data'].append(structure)
        id += 1
json.dump(dataset, d_file)
c_file.close()
d_file.close()