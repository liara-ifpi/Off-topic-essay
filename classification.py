import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from util import read_results
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

#pip install imbalanced-learn

training_corpus = read_results('resultados_training')
testing_corpus = read_results('resultados_testing')

X_test = testing_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y_test = testing_corpus['label']

X = training_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y = training_corpus['label']

ros = SMOTE(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(classification_report_imbalanced(y_test, y_pred))

print(f'Acurácia balanceada: {balanced_accuracy_score(y_test, y_pred)}')

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
# disp.plot()
# plt.show()
