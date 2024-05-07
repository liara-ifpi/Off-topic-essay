import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier,  BaggingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression,SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm, tree, linear_model
from sklearn.svm import SVC,  NuSVC,  LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid, NeighborhoodComponentsAnalysis
from util import read_results
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

#pip install imbalanced-learn

import numpy as np
np.random.seed(42)


training_corpus = read_results('resultados_training')
testing_corpus = read_results('resultados_testing')

X_test = testing_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y_test = testing_corpus['label']

X = training_corpus[['boolean', 'tf', 'tfidf', 'embeddings', 'st', 'wmd']]
y = training_corpus['label']

ros = SMOTE(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

clf = QuadraticDiscriminantAnalysis()

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
