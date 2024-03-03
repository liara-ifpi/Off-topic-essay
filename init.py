from math import e
import re
from statistics import StatisticsError, mean
import pandas as pd

from features import Features
from similarity import cosine_similarity
from util import read_corpus, read_nota, read_prompts


if __name__ == "__main__":
    # dataframes para os aruivos essay-br e prompts no diretório essays
    # notas_zero = read_notazero()
    notas = read_nota()
    #notas_todas = read_corpus()
    prompts = read_prompts()

    # Instancia a classe Features responsável por transformar uma sentença em um vetor
    # type=0 -> boolean features
    # type=1 -> TF features
    # type=2 -> TF-IDF features
    # type=3 -> Embeddings
    # type=4 -> Sentence Embeddings
    type = 3
    features = Features(type)

    # Criar um DataFrame com os resultados
    df_results = pd.DataFrame(columns=['similarity'])

    # Itera sobre todas as linhas do arquivo notazero
    for index, row in notas.iterrows():
        # pega o tópico das redações
        prompt_id = row['prompt']

        # mapeia o id do arquivo notazero com o id do arquivo prompts
        # prompt = prompts[prompts['id'] == prompt_id]
        # prompt = prompts.loc[prompts['id'] == prompt_id]
        description = prompts.loc[prompts['id'] == prompt_id, 'description'].item()

        # cria uma lista de parágrafos (texto motivacional) da coluna description do arquivo prompts
        regex = r"'(.*?)'"
        topic_paragraphs = re.findall(regex, description.replace('[', '').replace(']', ''))
        # print(topic_paragraphs)

        similarities = []  # Lista para armazenar as similaridades

        # calcula a similaridade de cada parágrafo do texto motivacional com cada parágrafo da redação
        for p in topic_paragraphs:
            for paragraph_essay in row['essay']:
                # for paragraph_essay in paragraphs_essay:
                if type == 3:
                    similarities.append(features.sent_to_vec(p, paragraph_essay, type, cos=True))
                elif type == 4:
                    similarities.append(features.sent_to_vec(p, paragraph_essay, 4))

                else:
                    snt1_vec, snt2_vec = features.sent_to_vec(p, paragraph_essay, 0)
                    similarity = cosine_similarity(snt1_vec, snt2_vec)
                    similarities.append(similarity)

        try:
            # calcula a média de similaridade
            df_results.loc[len(df_results)] = {'similarity': mean(similarities)}
        except StatisticsError:
            # verificar porque alguns valores estão zerados
            df_results.loc[len(df_results)] = {'similarity': 0}

    # Salvar o DataFrame em um arquivo CSV
    df_results.to_csv('resultados_development_emb.csv', index=False)
