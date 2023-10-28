import pandas as pd
import openpyxl

# Ler os dados da primeira planilha
df = pd.read_csv('diferenca/resultadoszero.csv')

# Calcular as métricas
diff_abs_media = df['similarity'].abs().mean()
diff_quadratica_media = (df['similarity'].diff()**2).mean()
diff_maxima = df['similarity'].diff().abs().max()

# Criar um DataFrame com os resultados
resultados = pd.DataFrame({
    'Diferença Absoluta Média': [diff_abs_media],
    'Diferença Quadrática Média': [diff_quadratica_media],
    'Diferença Máxima': [diff_maxima]
})

# Escrever os resultados em uma nova planilha
with pd.ExcelWriter('planilhazero.xlsx', engine='openpyxl') as writer:
    resultados.to_excel(writer, sheet_name='Resultados', index=False)

print("Resultados calculados e armazenados em 'planilha'")
