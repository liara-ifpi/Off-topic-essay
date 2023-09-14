from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd


class Features:

    def __init__(self, type) -> None:
        # boolean features
        if type == 0:
            self.vector = CountVectorizer(binary=True)
        
        # tf features
        elif type == 1:
            self.vector = TfidfVectorizer(use_idf=False)
        
        # tf-idf features
        else:
            self.vector = TfidfVectorizer()
    
    def sent_to_vec(self, snt1, snt2):
        snt_vector = self.vector.fit_transform([snt1, snt2])
        return snt_vector.toarray()

