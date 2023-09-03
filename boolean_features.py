from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine

import pandas as pd
import numpy as np


class BooleanFeatures:

    def __init__(self) -> None:
        self.vector = CountVectorizer(binary=True)
    
    def sent_to_vec(self, snt1, snt2):
        snt_vector = self.vector.fit_transform([snt1, snt2])
        return snt_vector.toarray()