import pandas as pd
import os, re

from boolean_features import BooleanFeatures
from similarity import cosine_similarity



def read_corpus_and_prompts():
    path = 'essays'
    essays = pd.read_csv(os.path.join(path, 'essay-br.csv'), converters={'essay': eval, 'competence': eval})
    prompts = pd.read_csv(os.path.join(path, 'prompts.csv'))
    return essays, prompts



if __name__ == "__main__":
    essays, prompts = read_corpus_and_prompts()
    boolean_features = BooleanFeatures()

    topic = essays['prompt'][0]
    result = prompts[prompts['id'] == topic]
    regex = r"'(.*?)'"
    for paragraph in result['description']:
        paragraph = paragraph.replace('[', '').replace(']', '')
        topic_paragraphs = re.findall(regex, paragraph)
    
    for p in topic_paragraphs:
        for essay in essays['essay'][0]:
            snt1_vec, snt2_vec = boolean_features.sent_to_vec(p, essay)
            print(cosine_similarity(snt1_vec, snt2_vec))
            
            
        
