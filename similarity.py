from scipy.spatial.distance import cosine


def cosine_similarity(snt1, snt2):
    return 1 - cosine(snt1, snt2)