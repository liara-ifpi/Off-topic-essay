import csv

def processar_csv(entrada_csv, saida_csv, coluna_analisada):
    dados_processados = []

    with open(entrada_csv, 'r') as arquivo_entrada:
        leitor_csv = csv.reader(arquivo_entrada)
        header = next(leitor_csv)
        indice_coluna = header.index(coluna_analisada)

        for linha in leitor_csv:
            nova_linha = processar_valores(linha, indice_coluna)
            dados_processados.append(nova_linha)

    with open(saida_csv, 'w', newline='') as arquivo_saida:
        escritor_csv = csv.writer(arquivo_saida)
        escritor_csv.writerow(header)
        escritor_csv.writerows(dados_processados)

def processar_valores(valores, indice_coluna):
    nova_linha = list(valores)
    valor_int = int(valores[indice_coluna])

    if valor_int != 0:
        novo_valor = 1
    else:
        novo_valor = 0

    nova_linha[indice_coluna] = novo_valor
    return nova_linha

# Exemplo de uso
entrada_csv = 'essays/development.csv'
saida_csv = 'results.csv'
coluna_analisada = 'score' 
processar_csv(entrada_csv, saida_csv, coluna_analisada)
