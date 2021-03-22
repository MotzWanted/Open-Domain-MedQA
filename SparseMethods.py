import math
import nltk
from nltk import FreqDist
from nltk.stem import SnowballStemmer
from tqdm import tqdm
import pickle
import numpy as np
stemmer = SnowballStemmer("english")

import os
dirname = os.path.dirname(__file__)

class SparseMethods:
    def __init__(self, Document_Frequency_Distributions, avgQueryLen, avgDocLen, ngram_func):
        self.ngram_func = ngram_func
        self.avgQueryLen = avgQueryLen
        self.avgDocLen = avgDocLen
        self.Doc_Freq_Dists = Document_Frequency_Distributions

    # static helper functions for n-grams used by MedQA
    @staticmethod
    def gen_unigrams(tokens):
        return [(token) for token in tokens]

    @staticmethod
    def gen_bigrams(tokens):
        return nltk.bigrams(tokens)

    @staticmethod
    def gen_trigrams(tokens):
        return nltk.trigrams(tokens)

    @staticmethod
    def gen_skipbigrams(tokens):
        return nltk.skipgrams(tokens, 2, 1)
    
    @staticmethod
    def prep_query(string):
        tokenized = nltk.word_tokenize(string)
        return [stemmer.stem(e) for e in tokenized]

    @staticmethod
    def prep_document(string):
        tokenized = nltk.word_tokenize(string)
        return [stemmer.stem(e) for e in tokenized]
    
    @staticmethod
    def ngram_freq_dist(tokens, ngram_func):
        return nltk.FreqDist(ngram_func(tokens))
    
    @staticmethod
    def snowball_stem(tokens):
        return stemmer.stem(tokens)
    
    def f(self, q_i, QorD):
        # from a n-gram frequency distribution return q_i frequency if any
        return QorD[q_i]

    def IDF(self, q_i):
        N = len(self.Doc_Freq_Dists)
        df_t = 0  # documents containing the n-gram
        for _, freq_dist in enumerate(self.Doc_Freq_Dists):
            if q_i in freq_dist:
                df_t += 1
                continue
        return math.log(N/(df_t+1))

    def S(self, Q, D):
        queryLen = len(Q)
        docLen = len(D)
        Q = nltk.FreqDist(self.ngram_func(Q))
        D = nltk.FreqDist(self.ngram_func(D))
        # hyper-parameters (USMLE)
        k_Q = 0.40
        b_Q = 0.70
        k_D = 0.90
        b_D = 0.35
        results = []
        for q_i, _ in Q.items():
            # terms
            idf = self.IDF(q_i)
            f_q = self.f(q_i, Q)
            f_d = self.f(q_i, D)
            # compute bm25 from equation defined by the MedQA paper
            bm25 = (idf * f_d * (k_D + 1)) / (f_q + k_D *
                                              (1 - b_D + (b_D * (docLen / self.avgDocLen))))
            # compute intermidiate s result
            s = (bm25 * idf * f_q * (k_Q + 1)) / (f_q + k_Q *
                                                  (1 - b_Q + b_Q * (queryLen / self.avgQueryLen)))
            results.append(s)
        return sum(results)