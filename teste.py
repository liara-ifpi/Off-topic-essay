from sklearn import svm
import pandas as pd

# Carregar os dados
X = pd.read_csv('essays/essays-results-x.csv')
y = pd.read_csv('essays/essays-results-y.csv')
print(y)
# Criar uma instância do classificador SVM
clf = svm.SVC(kernel='linear')

# Treinar o modelo SVM
clf.fit(X, y)

# Prever os rótulos das classes para os dados de entrada X
y_pred = clf.predict(X)

# Adicionar as previsões como uma nova coluna aos dados originais X
X['predicted_labels'] = y_pred

# Salvar os dados com as previsões em um novo arquivo CSV
X.to_csv('essays-results-with-predictions.csv', index=False)
