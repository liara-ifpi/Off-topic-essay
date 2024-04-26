
import re
from statistics import StatisticsError, mean
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import Features
from similarity import cosine_similarity
from util import read_corpus, read_set, read_prompts
from nltk.tokenize import sent_tokenize


if __name__ == "__main__":
    # dataframes para os aruivos essay-br e prompts no diretório essays
    # notas_zero = read_notazero()
    prompts = read_prompts()
    corpus = ['testing', 'development', 'training']
    for c in corpus:
        notas = read_set(c)
        #notas_todas = read_corpus()
    
        # Instancia a classe Features responsável por transformar uma sentença em um vetor
        # type=0 -> boolean features
        # type=1 -> TF features
        # type=2 -> TF-IDF features
        # type=3 -> Embeddings
        # type=4 -> Sentence Embeddings
        # type = 0
        features = Features()

        # Criar um DataFrame com os resultados
        df_results = pd.DataFrame(columns=['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd', 'label'])

        # Itera sobre todas as linhas do arquivo notazero
        for index, row in tqdm(notas.iterrows(), desc='Extracting features from ' + c):
            # pega o tópico das redações
            prompt_id = row['prompt']
            label = row['score']
            if label > 0:
                label = 1

            # mapeia o id do arquivo notazero com o id do arquivo prompts
            # prompt = prompts[prompts['id'] == prompt_id]
            # prompt = prompts.loc[prompts['id'] == prompt_id]
            description = prompts.loc[prompts['id'] == prompt_id, 'description'].item()

            # cria uma lista de parágrafos (texto motivacional) da coluna description do arquivo prompts
            # regex = r"'(.*?)'"
            # topic_paragraphs = re.findall(regex, description.replace('[', '').replace(']', ''))
            topic_paragraphs = sent_tokenize(description, language='portuguese')
            # print(topic_paragraphs)

            similarities = []  # Lista para armazenar as similaridades

            # calcula a similaridade de cada parágrafo do texto motivacional com cada parágrafo da redação
            for p in topic_paragraphs:
                for paragraph_essay in row['essay']:
                    values = []
                    # for paragraph_essay in paragraphs_essay:
                    for i in range(5):
                        values.append(features.sent_to_vec(p, paragraph_essay, i))
                    values.append(features.sent_to_vec(p, paragraph_essay, 3, False))
                    similarities.append(values)
                    # if type == 3:
                    #     similarities.append(features.sent_to_vec(p, paragraph_essay, type, cos=True))
                    # elif type == 4:
                    #     similarities.append(features.sent_to_vec(p, paragraph_essay, 4))

                    # else:
                    #     snt1_vec, snt2_vec = features.sent_to_vec(p, paragraph_essay, 0)
                    #     similarity = cosine_similarity(snt1_vec, snt2_vec)
                    #     similarities.append(similarity)

            try:
                # calcula a média de similaridade
                # df_results.loc[len(df_results)] = {'similarity': mean(similarities)}
                avg = np.mean(similarities, axis=0)
                dict = {'boolean': avg[0], 'tf': avg[1], 'tfidf': avg[2], 'embeddings': avg[3], 'st': avg[4], 'wmd': avg[5], 'label': label}
                df_results.loc[len(df_results)] = dict
            except StatisticsError as e:
                print(e)
                # verificar porque alguns valores estão zerados
                # df_results.loc[len(df_results)] = {'similarity': 0}

        # Salvar o DataFrame em um arquivo CSV
        # df_results = pd.DataFrame(dict)
        df_results.to_csv('resultados_'+c+'.csv', index=False)
