### SMOTE = GradientBoostingClassifier

*melhor resultado entre os usados*

 precision    recall  f1-score   support

    0       0.02      0.18      0.04        11
    1       0.98      0.87      0.92       675

    accuracy                           0.85       686
   macro avg       0.50      0.52      0.48       686
weighted avg       0.97      0.85      0.91       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.02      0.18      0.87      0.04      0.40      0.15        11
    1       0.98      0.87      0.18      0.92      0.40      0.17       675

avg / total       0.97      0.85      0.19      0.91      0.40      0.17       686

Acurácia balanceada: 0.5235016835016835
[[  2   9]
 [ 91 584]]

### RandomOverSampler = GradientBoostingClassifier

 precision    recall  f1-score   support

    0       0.00      0.00      0.00        11
    1       0.98      0.97      0.98       675

    accuracy                           0.95       686
   macro avg       0.49      0.48      0.49       686
weighted avg       0.97      0.95      0.96       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.00      0.00      0.97      0.00      0.00      0.00        11
    1       0.98      0.97      0.00      0.98      0.00      0.00       675

avg / total       0.97      0.95      0.02      0.96      0.00      0.00       686

Acurácia balanceada: 0.48444444444444446
[[  0  11]
 [ 21 654]]

### ADASYN = GradientBoostingClassifier

precision    recall  f1-score   support

    0       0.01      0.09      0.02        11
    1       0.98      0.87      0.92       675

    accuracy                           0.86       686
   macro avg       0.50      0.48      0.47       686
weighted avg       0.97      0.86      0.91       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.09      0.87      0.02      0.28      0.07        11
    1       0.98      0.87      0.09      0.92      0.28      0.09       675

avg / total       0.97      0.86      0.10      0.91      0.28      0.09       686

Acurácia balanceada: 0.48175084175084176
[[  1  10]
 [ 86 589]]


# svm

    precision    recall  f1-score   support

    0       0.01      0.09      0.01        11
    1       0.98      0.80      0.88       675

    accuracy                           0.79       686
   macro avg       0.49      0.45      0.45       686
weighted avg       0.97      0.79      0.87       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.09      0.80      0.01      0.27      0.07        11
    1       0.98      0.80      0.09      0.88      0.27      0.08       675

avg / total       0.97      0.79      0.10      0.87      0.27      0.08       686

Acurácia balanceada: 0.4476767676767677
[[  1  10]
 [132 543]]

# Perceptron

 precision    recall  f1-score   support

    0       0.02      0.82      0.04        11
    1       0.99      0.27      0.43       675

    accuracy                           0.28       686
   macro avg         0.50      0.55      0.23       686
weighted avg       0.97      0.28      0.42       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.02      0.82      0.27      0.04      0.47      0.24        11
    1       0.99      0.27      0.82      0.43      0.47      0.21       675

avg / total       0.97      0.28      0.81      0.42      0.47      0.21       686

Acurácia balanceada: 0.5453872053872054
[[  9   2]
 [491 184]]



# SGDClassifier

 precision    recall  f1-score   support

    0       0.00      0.00      0.00        11
           1       0.98      0.93      0.96       675

    accuracy                           0.91       686
   macro avg       0.49      0.46      0.48       686
weighted avg       0.97      0.91      0.94       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.00      0.00      0.93      0.00      0.00      0.00        11
          1       0.98      0.93      0.00      0.96      0.00      0.00       675

avg / total       0.97      0.91      0.01      0.94      0.00      0.00       686

Acurácia balanceada: 0.46444444444444444
[[  0  11]
 [ 48 627]]



# 1.9. Naive Bayes[¶](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes "Link to this heading")

## GaussianNB

precision    recall  f1-score   support

    0       0.01      0.18      0.02        11
           1       0.98      0.69      0.81       675

    accuracy                           0.68       686
   macro avg       0.50      0.44      0.41       686
weighted avg       0.97      0.68      0.80       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.18      0.69      0.02      0.35      0.12        11
          1       0.98      0.69      0.18      0.81      0.35      0.13       675

avg / total       0.97      0.68      0.19      0.80      0.35      0.13       686

Acurácia balanceada: 0.4368350168350168
[[  2   9]
 [208 467]]

## BernoulliNB

 precision    recall  f1-score   support

    0       0.02      1.00      0.03        11
           1       1.00      0.00      0.01       675

    accuracy                           0.02       686
   macro avg       0.51      0.50      0.02       686
weighted avg       0.98      0.02      0.01       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.02      1.00      0.00      0.03      0.07      0.00        11
          1       1.00      0.00      1.00      0.01      0.07      0.00       675

avg / total       0.98      0.02      0.98      0.01      0.07      0.00       686

Acurácia balanceada: 0.5022222222222222
[[ 11   0]
 [672   3]]


# 1.10. Decision Trees

## 1.10.1. Classification

 precision    recall  f1-score   support

    0       0.01      0.09      0.02        11
    1       0.98      0.88      0.93       675

    accuracy                           0.87       686
   macro avg       0.50      0.48      0.47       686
weighted avg       0.97      0.87      0.91       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.09      0.88      0.02      0.28      0.07        11
    1       0.98      0.88      0.09      0.93      0.28      0.09       675

avg / total       0.97      0.87      0.10      0.91      0.28      0.09       686

Acurácia balanceada: 0.4847138047138047
[[  1  10]
 [ 82 593]]

# 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking[¶](https://scikit-learn.org/stable/modules/ensemble.html#ensembles-gradient-boosting-random-forests-bagging-voting-stacking "Link to this heading")

## 1.11.1. HistGradientBoostingClassifier

 precision    recall  f1-score   support

    0       0.00      0.00      0.00        11
    1       0.98      0.92      0.95       675

    accuracy                           0.90       686
   macro avg       0.49      0.46      0.47       686
weighted avg       0.97      0.90      0.93       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.00      0.00      0.92      0.00      0.00      0.00        11
    1       0.98      0.92      0.00      0.95      0.00      0.00       675

avg / total       0.97      0.90      0.01      0.93      0.00      0.00       686

Acurácia balanceada: 0.4585185185185185
[[  0  11]
 [ 56 619]]

## 1.11. RandomForestClassifier

   precision    recall  f1-score   support

    0       0.00      0.00      0.00        11
    1       0.98      0.96      0.97       675

    accuracy                           0.94       686
   macro avg       0.49      0.48      0.49       686
weighted avg       0.97      0.94      0.96       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.00      0.00      0.96      0.00      0.00      0.00        11
    1       0.98      0.96      0.00      0.97      0.00      0.00       675

avg / total       0.97      0.94      0.02      0.96      0.00      0.00       686

Acurácia balanceada: 0.48
[[  0  11]
 [ 27 648]]


## [1.11.3. Bagging meta-estimator]()

precision    recall  f1-score   support

    0       0.01      0.09      0.02        11
           1       0.98      0.89      0.94       675

    accuracy                           0.88       686
   macro avg       0.50      0.49      0.48       686
weighted avg       0.97      0.88      0.92       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.09      0.89      0.02      0.28      0.07        11
          1       0.98      0.89      0.09      0.94      0.28      0.09       675

avg / total       0.97      0.88      0.10      0.92      0.28      0.09       686

Acurácia balanceada: 0.4921212121212121
[[  1  10]
 [ 72 603]]

# 1.16. Probability calibration[¶](https://scikit-learn.org/stable/modules/calibration.html#probability-calibration "Link to this heading")


# 1.17. Neural network models (supervised)[¶](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-network-models-supervised "Link to this heading")



## 1.17.2. Classification

precision    recall  f1-score   support

    0       0.01      0.18      0.02        11
    1       0.98      0.67      0.80       675

    accuracy                           0.66       686
   macro avg       0.49      0.43      0.41       686
weighted avg       0.96      0.66      0.79       686

    pre       rec       spe        f1       geo       iba       sup

    0       0.01      0.18      0.67      0.02      0.35      0.12        11
    1       0.98      0.67      0.18      0.80      0.35      0.13       675

avg / total       0.96      0.66      0.19      0.79      0.35      0.13       686

Acurácia balanceada: 0.42720538720538725
[[  2   9]
 [221 454]]
