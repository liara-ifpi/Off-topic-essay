from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors

from util import preprocess
from scipy import spatial

import numpy as np


class Features:

    def __init__(self, type) -> None:
        # boolean features
        if type == 0:
            self.vector = CountVectorizer(binary=True)
        
        # tf features
        elif type == 1:
            self.vector = TfidfVectorizer(use_idf=False)
        
        # tf-idf features
        elif type == 2:
            self.vector = TfidfVectorizer()
        
        else:
            # model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
            # model.save(path)
            # pip install gensim
            self.vector = KeyedVectors.load('embedding/embeddings', mmap='r')
            
    
    def sent_to_vec(self, snt1, snt2, type):
        if type == 3:
            tokens1, tokens2 = preprocess(snt1, snt2)
            return self.vector.wmdistance(tokens1, tokens2)
            
        else:
            snt_vector = self.vector.fit_transform([snt1, snt2])
            return snt_vector.toarray()
    

    def cosine_distance_embeddings(self, tokens1, tokens2):
        vector1 = np.mean([self.vector[token] for token in tokens1], axis=0)
        vector2 = np.mean([self.vector[token] for token in tokens2], axis=0)
        return 1 - spatial.distance.cosine(vector1, vector2)

