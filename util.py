
import random
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
    return pd.read_csv(os.path.join(path, 'essay-br-2.csv'), converters={'essay': eval, 'competence': eval})


def read_results(corpus):
    return pd.read_csv(corpus+'.csv')

#def read_notazero():
   # '''Lê o arquivo notazero e retorna um dataframe'''
   # return pd.read_csv(os.path.join(path, 'notazero.csv'), converters={'essay': eval, 'competence': eval})



#def read_notamil():
  # '''Lê o arquivo notazero e retorna um dataframe'''
  # return pd.read_csv(os.path.join(path, 'notamil.csv'), converters={'essay': eval, 'competence': eval})

def read_set(corpus):
    '''Lê o arquivo notazero e retorna um dataframe'''
    return pd.read_csv(os.path.join(path, corpus+'.csv'), converters={'essay': eval, 'competence': eval})


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


def count_labels():

    df = read_results('balanced-training')
    print(df['score'].value_counts())


def replace_prompt(x):
    value = random.randint(0, 85)
    while value == x:
        value = random.randint(0, 85)
    return value


def balancing():
    # 1543 + 56 = 1599
    # numbers_of_zero = 82
    df = read_results('essays/training')
    # df1 = read_results('essay-on-topic')
    df_zeros = df[df['score'] == 0]

    df_ontopic = df[df['score'] > 0]
    df_sample = df_ontopic.sample(1543, random_state=42)

    df_offtopic = df_ontopic[df_ontopic.index.isin(df_sample.index) == False]
    # print(df_offtopic)


    # print(df_sample)
    df_sample['prompt'] = df_sample['prompt'].apply(replace_prompt)
    df_sample['score'] = df_sample['score'].fillna(0, inplace=True)
    # print(df_sample)



    # df1 = df.sample(4488)
    balanced = [df_zeros, df_sample, df_offtopic]
    result = pd.concat(balanced)

    result.to_csv('balanced-training.csv', index=False)



    # print(type(score))
    # prompt = df['prompt']


if __name__ == "__main__":
    count_labels()
