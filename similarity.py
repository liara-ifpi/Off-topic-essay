import numpy as np

from scipy.spatial.distance import cosine
from scipy import spatial


def cosine_similarity(snt1, snt2):
    return 1 - cosine(snt1, snt2)


def cosine_distance_embeddings(vector, tokens1, tokens2):
        vector1 = np.mean([vector[token] for token in tokens1 if token in vector.key_to_index], axis=0).ravel()
        vector2 = np.mean([vector[token] for token in tokens2 if token in vector.key_to_index], axis=0).ravel()
        return 1 - spatial.distance.cosine(vector1, vector2)