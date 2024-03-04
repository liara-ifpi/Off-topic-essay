import pandas as pd


# importacao de bibliotecas


# tabela = pd.read_excel("essay-br.xlsx", engine='openpyxl') # variavel recebe a planilha junto com outra biblioteca para leitura e modificaçao
tabela = pd.read_csv("essays/testing-results.csv", converters={'essay': eval, 'competence': eval}) # variavel recebe a planilha junto com outra biblioteca para leitura e modificaçao

filtrados = tabela['score'] == 1 # a variavel filtrados recebe a tabela com parametros 


pd.get_option('display.max_rows')
pd.set_option('display.max_row', 4000) # aumenta o numero de colunas a serem visualisadas

print("dados filtrados:")
print(tabela[filtrados])

# tabela[filtrados].to_excel('filtrados.xlsx', index=False) # salva e cria o artivro em xlsx
tabela[filtrados].to_csv('nota.csv', index=False) # salva e cria o artivro em csv