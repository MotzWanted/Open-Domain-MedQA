import sys
import json
from tqdm import std, tqdm
from os import getenv, path
import re
import spacy
import scispacy

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_sci_sm") # trained on UML tags and other huge biomedical libraries

# Create MedQA Dataset
# Command: cat qbank | python generate_medqa_dataset.py > medqa_dataset.json

# open file descriptors for 
# c_file = open("medqa_dataset_construction/questions/US/US_qbank.jsonl", 'r')

qbank = sys.stdin.read().splitlines()

dataset = {
    'version': '0.0.1',
    'data': []
}

# focus_words = {'is', 'does', 'do', 'what', 'when', 'where', 'who', 'why', 'what', 'how', 'which'}

id = 0
for line in tqdm(qbank):
    sample:json = json.loads(line)

    clean_sample_question = re.sub(r'\n+', '', sample['question'])
    doc = nlp(clean_sample_question)
    sents = list(doc.sents)

    question = sents[-1].text.strip()
    symptoms = ' '.join(t.text for t in sents[:-1]).strip()

    answer = sample['options'].get(sample['answer_idx']) # we may want to add more answers ex. using MetaMap
    options = sample['options']
    answer_idx = sample['answer_idx']

    structure = {
            'id': id,
            #'title': # we don't have title of article or section where the answer was given...
            'symptoms': symptoms, # patient symptoms
            'question': question,
            'answer': answer,
            'options': options,
            'answer_idx': answer_idx
    }
    dataset['data'].append(structure)
    id += 1

json.dump(dataset, sys.stdout, indent=2)