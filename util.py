from nltk import tokenize
from nltk.corpus import stopwords
import pandas as pd
import os


path = 'essays'

def read_prompts():
    '''Lê o arquivo prompts e retorna um dataframe'''
    return pd.read_csv(os.path.join(path, 'prompts.csv'))


def read_corpus():
    '''Lê o arquivo essay-br e retorna um dataframe'''
    return pd.read_csv(os.path.join(path, 'essay-br.csv'), converters={'essay': eval, 'competence': eval})


def read_notazero():
    '''Lê o arquivo notazero e retorna um dataframe'''
    return pd.read_csv(os.path.join(path, 'notazero.csv'), converters={'essay': eval, 'competence': eval})



#def read_notamil():
   # '''Lê o arquivo notazero e retorna um dataframe'''
   # return pd.read_csv(os.path.join(path, 'notamil.csv'), converters={'essay': eval, 'competence': eval})


#def read_nota():
   # '''Lê o arquivo notazero e retorna um dataframe'''
   # return pd.read_csv(os.path.join(path, 'similariedades01.csv'), converters={'essay': eval, 'competence': eval})


def preprocess(snt1, snt2):
    '''Tokeniza as sentenças e retorna a lista de tokens
    pip install nltk
    
    python
    import nltk
    download('punk')
    download('stopwords')
    '''
    tokens1 = tokenize.word_tokenize(snt1, language='portuguese')
    tokens2 = tokenize.word_tokenize(snt2, language='portuguese')
    return [t.lower() for t in tokens1 if t not in stopwords.words(u'portuguese')], [t.lower() for t in tokens2 if t not in stopwords.words(u'portuguese')],