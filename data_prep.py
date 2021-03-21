from nltk.tokenize import sent_tokenize, word_tokenize
from os import listdir
from os.path import isfile, join
import pickle

output_folder = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/textbooks/en_sentence/"
data_path = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/textbooks/en/"
textbooks = [f for f in listdir(data_path) if isfile(join(data_path, f))]

for book in textbooks:
    with open(data_path + book, 'r') as file:
        content = file.read().replace('\n', '')
    
    sentences = sent_tokenize(content)

    with open(output_folder + book.replace('.txt','') + '.pkl', 'wb') as f:
        pickle.dump(sentences, f)




with open(data_path + 'Physiology_Levy.txt', 'r') as file:
        content = file.read().replace('\n', '')

sentences = sent_tokenize(content)

with open(output_folder + book.replace('.txt','') + '.pkl', 'wb') as f:
    pickle.dump(sentences, f)

with open(output_folder + 'Physiology_Levy.pkl',"rb") as file:
    test = pickle.load(file)