

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from util import read_results
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


training_corpus = read_results('resultados_training')
testing_corpus = read_results('resultados_testing')

X_test = testing_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y_test = testing_corpus['label']

X = training_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y = training_corpus['label']

ros = SMOTE(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

clf = GradientBoostingClassifier()
clf.fit(X_resampled, y_resampled)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(f1_score(y_test, y_pred, average='weighted'))

print(balanced_accuracy_score(y_test, y_pred))