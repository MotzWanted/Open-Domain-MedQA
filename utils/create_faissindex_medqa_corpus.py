import torch
import sys
import json
from os import getenv, path, listdir
from google.colab import drive
from datasets import load_dataset
from transformers import BertModel, BertTokenizer, PreTrainedTokenizerFast, AutoTokenizer, AutoModel
from tokenizers.processors import TemplateProcessing

CACHE_DIR = "/cache/"

# load server google drive
# request access from Benjamin Starostka
drive.mount('/content/data', force_remount=True)

# constants
gdrive_data = 'data/MyDrive/Shared/'
gdrive_data_medqa = 'data/MyDrive/Shared/MedQA/corpus/'


#### Create FAISS Index from MedQA Corpus ####

# configuration and loading step
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

special_tokens = '[D]'
tokenizer.add_tokens([special_tokens])

books = ['Anatomy_Gray.json',
         'Biochemistry_Lippincott.json',
         'Cell_Biology_Alberts.json',
         'First_Aid_Step1.json',
         'First_Aid_Step2.json',
         'Gynecology_Novak.json',
         'Histology_Ross.json',
         'Immunology_Janeway.json',
         'InternalMed_Harrison.json',
         'Neurology_Adams.json',
         'Obstentrics_Williams.json',
         'Pathology_Robbins.json',
         'Pathoma_Husain.json',
         'Pediatrics_Nelson.json',
         'Pharmacology_Katzung.json',
         'Physiology_Levy.json',
         'Psichiatry_DSM-5.json',
         'Surgery_Schwartz.json']
books = ['Anatomy_Gray.json'] # Debug: single override
corpus = load_dataset('json', data_files=[path.join(gdrive_data_medqa, bookname) for bookname in books], field='data') # split='train[:100%]'

# begin FAISS indexing and write to file
torch.set_grad_enabled(False)

corpus_embeddings = corpus.map(
        lambda example: {
            'embeddings': model(**tokenizer('[D]' + example['sentence'], max_length=200, padding=True, return_tensors='pt'))['pooler_output'][0].numpy()})
corpus_embeddings.save_to_disk(path.join(CACHE_DIR, "corpus/"))

corpus_embeddings.add_faiss_index(column='corpus', device=0, metric_type=0)

corpus_embeddings.save_faiss_index("corpus", path.join(gdrive_data_medqa, "medqa_corpus.faiss"))

