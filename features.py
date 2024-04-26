from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors

from similarity import cosine_similarity, cosine_distance_embeddings
from util import read_corpus, preprocess

from sentence_transformers import SentenceTransformer, util


class Features:

    def __init__(self) -> None:
        # boolean features
        # if type == 0:
        self.count_vector = CountVectorizer(binary=True)
        
        # tf features
        # elif type == 1:
        self.tf_vector = TfidfVectorizer(use_idf=False)
        
        # tf-idf features
        # elif type == 2:
        self.tfidf_vector = TfidfVectorizer()
        
        # elif type == 3:
            # model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')
            # model.save(path)
            # pip install gensim
        self.embeddings_vector = KeyedVectors.load('embedding/embeddings', mmap='r')
        
        # else:
            # pip install sentence-transformers
        self.st_vector = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            
    
    def sent_to_vec(self, snt1, snt2, type, cos=True):

        if type == 0:
            if not hasattr(self.count_vector, 'vocabulary_'):
                self.count_vector.fit_transform([snt1, snt2])
            snt1_vector, snt2_vector = self.count_vector.transform([snt1, snt2]).toarray()
            return cosine_similarity(snt1_vector, snt2_vector)
        elif type == 1:
            if not hasattr(self.tf_vector, 'vocabulary_'):
                self.tf_vector.fit_transform([snt1, snt2])
            snt1_vector, snt2_vector = self.tf_vector.transform([snt1, snt2]).toarray()
            return cosine_similarity(snt1_vector, snt2_vector)
        elif type == 2:
            if not hasattr(self.tfidf_vector, 'vocabulary_'):
                self.tfidf_vector.fit_transform([snt1, snt2])
            snt1_vector, snt2_vector = self.tfidf_vector.transform([snt1, snt2]).toarray()
            return cosine_similarity(snt1_vector, snt2_vector)
        elif type == 3:
            tokens1, tokens2 = preprocess(snt1, snt2)
            if cos:
                return cosine_distance_embeddings(self.embeddings_vector, tokens1, tokens2)
            else:
                return self.embeddings_vector.wmdistance(tokens1, tokens2)
        elif type == 4:
            
            embeddings_stn1 = self.st_vector.encode(snt1, convert_to_tensor=True)
            embeddings_stn2 = self.st_vector.encode(snt2, convert_to_tensor=True)
            return util.cos_sim(embeddings_stn1, embeddings_stn2)[0][0].item()
        

