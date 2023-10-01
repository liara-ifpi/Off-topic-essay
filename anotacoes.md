# **anotações**

* pelo que entendi o *resultadosinit.xlsx* é o resultado das similariedades(certo?), mas faltou o prompt para a identicarção da notas delas

* criar mais uma coluna para uso de identificar para o motivo das redações terem obtido zero

* para a comparação eu pensei em usar os resultados e joga-los em um modelo de maquina com `scikit-learn `
* li algumas coisas por ai e cheguei a pensar sobre os possiveis metados abaixo:

1. carregamos os os resultados que contém os pares de texto e as pontuações de similaridade de cosseno. (prompt, score e motivo de zerar )
2. dividimos os dados em conjuntos de treinamento e teste.(para isso temos que escolher um modelo)
3. usamos o `TfidfVectorizer` para extrair características dos textos.(acho interessente)

