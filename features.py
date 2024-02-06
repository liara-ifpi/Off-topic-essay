from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors

from util import read_corpus, preprocess
from scipy import spatial
from sentence_transformers import SentenceTransformer, util

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
        
        elif type == 3:
            # model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
            # model.save(path)
            # pip install gensim
            self.vector = KeyedVectors.load('embedding/embeddings', mmap='r')
        
        else:
            # pip install sentence-transformers
            self.vector = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            
    
    def sent_to_vec(self, snt1, snt2, type, cos=None):
        if type == 3:
            tokens1, tokens2 = preprocess(snt1, snt2)
            if cos:
                return self.cosine_distance_embeddings(tokens1, tokens2)
            else:
                if isinstance(self.vector, (CountVectorizer, TfidfVectorizer)):
                    if not hasattr(self.vector, 'vocabulary_'):
                        self.vector.fit_transform([snt1, snt2])
                return self.vector.wmdistance(tokens1, tokens2)
        elif type == 4:
            
            embeddings_stn1 = self.vector.encode(snt1, convert_to_tensor=True)
            embeddings_stn2 = self.vector.encode(snt2, convert_to_tensor=True)
            return util.cos_sim(embeddings_stn1, embeddings_stn2)[0][0].item()
            
        else:
            if isinstance(self.vector, (CountVectorizer, TfidfVectorizer)):
                if not hasattr(self.vector, 'vocabulary_'):
                    self.vector.fit_transform([snt1, snt2])
                snt_vector = self.vector.transform([snt1, snt2]).toarray()
                return snt_vector
    

    def cosine_distance_embeddings(self, tokens1, tokens2):
        vector1 = np.mean([self.vector[token] for token in tokens1 if token in self.vector.key_to_index], axis=0).ravel()
        vector2 = np.mean([self.vector[token] for token in tokens2 if token in self.vector.key_to_index], axis=0).ravel()
        return 1 - spatial.distance.cosine(vector1, vector2)

