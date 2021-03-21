import math
import nltk
from nltk import FreqDist
from tqdm import tqdm
import pickle


# helper functions for n-grams used by MedQA
def gen_bigrams(tokens):
    # returns a tuple
    return nltk.bigrams(tokens)


def gen_trigrams(tokens):
    return nltk.trigrams(tokens)


def gen_skipbigrams(tokens):
    return nltk.skipgrams(tokens, 2, 1)


# IR functions
def f(q_i, Q_or_D_FreqDist):
    # from a n-gram frequency distribution return q_i frequency if any
    return Q_or_D_FreqDist[q_i]


def IDF(q_i):
    # takes in an n-gram tuple
    N = len(q_i)
    # instantiate a dictionary of frequencies
    dictionary = dict()
    for word in q_i:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    # apply idf from frequency
    for word, val in dictionary.items():
        dictionary[word] = math.log(N / float(val))
    return dictionary


def s(Q, D):
    # Q is a collection of query n-grams
    # D is a n-gram frequency distributions
    # Note: D should be precomputed

    # hyper-parameters (USMLE)
    k_Q = 0.40
    b_Q = 0.70
    k_D = 0.90
    b_D = 0.35

    results = []
    for q_i in tqdm(Q):
        # terms
        idf = IDF(q_i)
        f_q = f(q_i, Q)
        f_d = f(q_i, D)
        queryLen = len(q_i)
        docLen = len(D[q_i])  # freq of q_i n-gram in Document
        doc_samples = []
        for ngram, freq in D.items():
            doc_samples.append(freq)
        avgDocLen = sum(doc_samples) / len(doc_samples)
        q_samples = []
        for ngram, freq in Q.items():
            q_samples.append(freq)
        avgQueryLen = sum(q_samples) / len(q_samples)

        # compute bm25 from equation defined by the MedQA paper
        bm25 = (IDF(q_i) * f(q_i, D) * (k_D + 1)) / (f(q_i, Q) +
                                                     k_D * (1 - b_D + (b_D * (docLen / avgDocLen))))

        # compute intermidiate s result
        s = (bm25 * idf * f_q * (k_Q + 1)) / (f_q + k_Q *
                                              (1 - b_Q + b_Q * (queryLen / avgQueryLen)))
        results.append(s)

    return sum(results)


data_path = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/dicts/"
output_folder = "/Users/andreasmotz/Documents/DTU/Kandidat/4. Semester/MEDQA/data_clean/textbooks/en_sentence/"
def benchmark():
    train = ""
    phys = ""
    with open(data_path + '/train.pickle',"rb") as file:
        train = pickle.load(file)
    with open(output_folder + 'Physiology_Levy.pkl',"rb") as file:
        phys = pickle.load(file)
    
    Q = nltk.bigrams(train["q1"]["qst"])
    print(Q)
    D = nltk.FreqDist(nltk.bigrams(phys))

    print(s(Q,D))

benchmark()