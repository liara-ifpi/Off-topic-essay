import re
import pandas as pd

from features import Features
from similarity import cosine_similarity
from util import read_notazero, read_prompts


if __name__ == "__main__":
    # dataframes para os aruivos essay-br e prompts no diretório essays
    notas_zero = read_notazero()
    prompts = read_prompts()

    # Instancia a classe Features responsável por transformar uma sentença em um vetor
    # type=0 -> boolean features
    # type=1 -> TF features
    # type=2 -> TF-IDF features
    features = Features(type=2)

    # Itera sobre todas as linhas do arquivo notazero
    for index, row in notas_zero.iterrows():
        # pega o tópico das redações
        prompt_id = row['prompt']

        # mapeia o id do arquivo notazero com o id do arquivo prompts
        prompt = prompts[prompts['id'] == prompt_id]

        # cria uma lista de parágrafos (texto motivacional) da coluna description do arquivo prompts
        regex = r"'(.*?)'"
        for paragraph in prompt['description']:
            paragraph = paragraph.replace('[', '').replace(']', '')
            topic_paragraphs = re.findall(regex, paragraph)

        similarities = []  # Lista para armazenar as similaridades

        # calcula a similaridade cada parágrafo do texto motivacional com cada parágrafo da redação
        for p in topic_paragraphs:
            for paragraphs_essay in notas_zero['essay']:
                for paragraph_essay in paragraphs_essay:
                    snt1_vec, snt2_vec = features.sent_to_vec(p, paragraph_essay)
                    similarity = cosine_similarity(snt1_vec, snt2_vec)
                    similarities.append(similarity)
    
    # Criar um DataFrame com os resultados
    df_results = pd.DataFrame({'Similarity': similarities})
    
    # Salvar o DataFrame em um arquivo CSV
    df_results.to_csv('resultadosinit.csv', index=False)
