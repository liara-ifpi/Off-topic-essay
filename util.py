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
