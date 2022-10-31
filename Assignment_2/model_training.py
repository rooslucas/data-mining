from lib2to3.pgen2.pgen import DFAState
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# TODO tune min_df values as well to see which is the best
# mention tfidf featur
X_train = pd.read_csv('Assignment 2/train_set.csv').iloc[:, 1]
Y_train = pd.read_csv('Assignment 2/train_labels.csv').iloc[:, 1]
X_test = pd.read_csv('Assignment 2/test_set.csv').iloc[:, 1]
Y_test = pd.read_csv('Assignment 2/test_labels.csv').iloc[:, 1]

cv_unigram = CountVectorizer(lowercase=True, ngram_range=(1, 1))
cv_bigram = CountVectorizer(lowercase=True, ngram_range=(1, 2))


# NOTE: removing stopwords seems to decrease performance!!!!!

# TODO cache values to speed up grid search!

# Function for confusion matrix
def display_confusion_matrix(y_true, y_pred):
    # Display confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred).ravel()
    print(
        f'True Negatives: {tn}, \nFalse Positives: {fp}, \nFalse Negatives: {fn}, \nTrue Positives: {tp}')

    precision = tp / (tp + fp)
    print(f'\nPrecision: {precision}')

    recall = tp / (tp + fn)
    print(f'Recall: {recall}')

    f1_score = 2 * ((precision*recall)/(precision + recall))
    print(f'F1 score: {f1_score}')
    # PPV_1 = tp / (tp + fp)
    # NPV_1 = tn / (tn + fn)

    # print(f'\nPPV: {PPV_1}')
    # print(f'NPV: {NPV_1}')


def make_predictions(model, param_grid, model_name):

    # make separate param grids for unigrams and bigrams
    param_grid_unigram = deepcopy(param_grid)
    param_grid_bigram = deepcopy(param_grid)

    # add hyperparameters for countvectorizers
    param_grid_unigram.update(
        {'cv_unigram__min_df': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    param_grid_bigram.update(
        {'cv_bigram__min_df': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})

    # add hyperparameters for best k mutual info features
    #param_grid_unigram.update({'best_mutual_info__k': [150, 200, 250, 300, 350]})
    #param_grid_bigram.update({'best_mutual_info__k': [150, 200, 250, 300, 350]})

    # pipeline_unigram = Pipeline([('cv_unigram', cv_unigram), ('best_mutual_info', SelectKBest(mutual_info_classif))\
    #                              , (model_name, model)])

    pipeline_unigram = Pipeline(
        [('cv_unigram', cv_unigram), (model_name, model)])

    # pipeline_bigram = Pipeline([('cv_bigram', cv_bigram),('best_mutual_info', \
    # SelectKBest(mutual_info_classif)), (model_name, model)])

    pipeline_bigram = Pipeline([('cv_bigram', cv_bigram), (model_name, model)])

    # implement grid seach for all hyperparameters
    clf_unigram = GridSearchCV(pipeline_unigram, param_grid=param_grid_unigram,
                               cv=10, verbose=True)

    clf_bigram = GridSearchCV(pipeline_bigram, param_grid=param_grid_bigram,
                              cv=10, verbose=True)

    best_clf_unigram = clf_unigram.fit(X_train, Y_train)
    print(
        f'The test accuracy for {model_name} on unigrams is: {best_clf_unigram.score(X_test, Y_test)}')
    print(best_clf_unigram.best_estimator_)
    print(best_clf_unigram.best_params_)
    y_pred = best_clf_unigram.predict(X_test)
    display_confusion_matrix(Y_test, y_pred)
    print()

    best_clf_bigram = clf_bigram.fit(X_train, Y_train)
    print(
        f'The test accuracy for {model_name} on unigrams and bigrams is: {best_clf_bigram.score(X_test, Y_test)}')
    print(best_clf_bigram.best_estimator_)
    print(best_clf_bigram.best_params_)
    y_pred = best_clf_bigram.predict(X_test)
    display_confusion_matrix(Y_test, y_pred)
    print()
    print()

    # elif model == lm:
    #     # features = best_clf_bigram.best_estimator_._final_estimator
    #     importances_uni = best_clf_unigram.best_estimator_._final_estimator.coef_
    #     indices_uni = np.argsort(importances_uni)[-10:]
    #     print("Feature importance Unigram:")
    #     for i in range(len(indices_uni)):
    #         print(f'{indices_uni[i]} : {importances_uni[indices_uni][i]}')

    #     importances_bi = best_clf_bigram.best_estimator_._final_estimator.coef_
    #     indices_bi = np.argsort(importances_bi)[-10:]
    #     print("\nFeature importance Bigram:")
    #     for i in range(len(indices_bi)):
    #         print(f'{indices_bi[i]} : {importances_bi[indices_bi][i]}')

    return best_clf_unigram, best_clf_bigram


############################################# Random Forest #######################################################
rf_name = 'random forest'
rf = RandomForestClassifier(random_state=37)
param_grid_rf = {f'{rf_name}__n_estimators': [100, 150, 200, 250, 300, 400], f'{rf_name}__max_depth': [
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], f'{rf_name}__max_features': ['auto', 'sqrt']}

rf_unigram, rf_bigram = make_predictions(rf, param_grid_rf, rf_name)

features = rf_unigram.best_estimator_[0].get_feature_names_out()
importances_uni = rf_unigram.best_estimator_._final_estimator.feature_importances_
indices_uni = np.argsort(importances_uni)[-10:]
print("Feature importance Unigram:")

for i in range(len(indices_uni)):
    print(
        f'{features[indices_uni][i]}, {indices_uni[i]} : {importances_uni[indices_uni[i]]} \n')

features_bi = rf_bigram.best_estimator_[0].get_feature_names_out()
importances_bi = rf_bigram.best_estimator_._final_estimator.feature_importances_
indices_bi = np.argsort(importances_bi)[-10:]
print("\nFeature importance Bigram:")

for i in range(len(indices_bi)):
    print(
        f'{features_bi[indices_bi][i]}, {indices_bi[i]} : {importances_bi[indices_bi[i]]} \n')

##################################################### Logstic Regression ######################################################
lm_name = 'logistic regression'
lm = LogisticRegression(penalty='l1', solver='liblinear', random_state=37)
param_grid_lm = {
    f'{lm_name}__C': np.logspace(-4, 4, 20), f'{lm_name}__max_iter': [100, 1000, 2500]}

lm_unigram, lm_bigram = make_predictions(lm, param_grid_lm, lm_name)

features = lm_unigram.best_estimator_[0].get_feature_names_out()
importances_uni = lm_unigram.best_estimator_._final_estimator.coef_
indices_uni = np.argsort(importances_uni)[-10:]
print("Feature importance Unigram:")

for i in range(len(indices_uni)):
    print(
        f'{features[indices_uni][i]}, {indices_uni[i]} : {importances_uni[indices_uni[i]]} \n')

features_bi = lm_bigram.best_estimator_[0].get_feature_names_out()
importances_bi = lm_bigram.best_estimator_._final_estimator.coef_
indices_bi = np.argsort(importances_bi)[-10:]
print("\nFeature importance Bigram:")

for i in range(len(indices_bi)):
    print(
        f'{features_bi[indices_bi][i]}, {indices_bi[i]} : {importances_bi[indices_bi[i]]} \n')

##################################################### Decision Tree ###########################################################
dt_name = 'decision tree'
dt = DecisionTreeClassifier(random_state=37)
param_grid_dt = {f'{dt_name}__max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], f'{dt_name}__min_impurity_decrease': [
    0.0, 0.001, 0.005, 0.01, 0.015, 0.02], f'{dt_name}__max_features': ['auto', 'sqrt']}

dt_unigram, dt_bigram = make_predictions(dt, param_grid_dt, dt_name)

features = dt_unigram.best_estimator_[0].get_feature_names_out()
importances_uni = dt_unigram.best_estimator_._final_estimator.feature_importances_
indices_uni = np.argsort(importances_uni)[-10:]
print("Feature importance Unigram:")

for i in range(len(indices_uni)):
    print(
        f'{features[indices_uni][i]}, {indices_uni[i]} : {importances_uni[indices_uni[i]]} \n')

features_bi = dt_bigram.best_estimator_[0].get_feature_names_out()
importances_bi = dt_bigram.best_estimator_._final_estimator.feature_importances_
indices_bi = np.argsort(importances_bi)[-10:]
print("\nFeature importance Bigram:")

for i in range(len(indices_bi)):
    print(
        f'{features_bi[indices_bi][i]}, {indices_bi[i]} : {importances_bi[indices_bi[i]]} \n')

######################################################### Naive Bayes #######################################################

mb_name = 'naive bayes'
mb = MultinomialNB()
param_grid_mb = {f'{mb_name}__alpha': [0.4]}  # , 0.5, 0.6, 0.7, 0.8, 1.0]}
mb_unigram, mb_bigram = make_predictions(mb, param_grid_mb, mb_name)

features = mb_unigram.best_estimator_[0].get_feature_names_out()
importances_uni = mb_unigram.best_estimator_._final_estimator.feature_log_prob_[
    1]
indices_uni = np.argsort(importances_uni)[-10:]
print("Feature importance Unigram:")

for i in range(len(indices_uni)):
    print(
        f'{features[indices_uni][i]}, {indices_uni[i]} : {importances_uni[indices_uni[i]]} \n')

importances_bi = mb_bigram.best_estimator_._final_estimator.feature_log_prob_[
    1]
indices_bi = np.argsort(importances_bi)[-10:]
print("\nFeature importance Bigram:")

for i in range(len(indices_bi)):
    features_bi = mb_bigram.best_estimator_[0].get_feature_names_out()
    print(
        f'{features_bi[indices_bi][i]}, {indices_bi[i]} : {importances_bi[indices_bi[i]]} \n')
