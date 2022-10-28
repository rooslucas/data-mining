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



#TODO tune min_df values as well to see which is the best
#mention tfidf featur

X_train = pd.read_csv('train_set.csv').iloc[:,1]
Y_train = pd.read_csv('train_labels.csv').iloc[:,1]
X_test = pd.read_csv('test_set.csv').iloc[:,1]
Y_test = pd.read_csv('test_labels.csv').iloc[:,1]

cv_unigram = CountVectorizer(lowercase=True, ngram_range = (1,1))
cv_bigram = CountVectorizer(lowercase=True, ngram_range = (1,2))


#NOTE: removing stopwords seems to decrease performance!!!!!

#TODO cache values to speed up grid search!


def make_predictions(model, param_grid, model_name):
    
    #make separate param grids for unigrams and bigrams
    param_grid_unigram = deepcopy(param_grid)
    param_grid_bigram = deepcopy(param_grid)
    
    #add hyperparameters for countvectorizers
    param_grid_unigram.update({'cv_unigram__min_df': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    param_grid_bigram.update({'cv_bigram__min_df': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    
    #add hyperparameters for best k mutual info features
    #param_grid_unigram.update({'best_mutual_info__k': [150, 200, 250, 300, 350]})
    #param_grid_bigram.update({'best_mutual_info__k': [150, 200, 250, 300, 350]})
    
    #pipeline_unigram = Pipeline([('cv_unigram', cv_unigram), ('best_mutual_info', SelectKBest(mutual_info_classif))\
    #                              , (model_name, model)])
        
    pipeline_unigram = Pipeline([('cv_unigram', cv_unigram) , (model_name, model)])
                                
    
    #pipeline_bigram = Pipeline([('cv_bigram', cv_bigram),('best_mutual_info', \
    # SelectKBest(mutual_info_classif)), (model_name, model)])
    
    pipeline_bigram = Pipeline([('cv_bigram', cv_bigram), (model_name, model)])
    
    #implement grid seach for all hyperparameters    
    clf_unigram = GridSearchCV(pipeline_unigram, param_grid=param_grid_unigram,
                          cv=10, verbose=True)
    
    clf_bigram = GridSearchCV(pipeline_bigram, param_grid=param_grid_bigram,
                          cv=10, verbose=True)
    
    best_clf_unigram = clf_unigram.fit(X_train, Y_train)
    print(f'The test accuracy for {model_name} on unigrams is: {best_clf_unigram.score(X_test, Y_test)}')
    print()
    
    best_clf_bigram = clf_bigram.fit(X_train, Y_train)
    print(f'The test accuracy for {model_name} on unigrams and bigrams is: {best_clf_bigram.score(X_test, Y_test)}')
    print()
    
    return best_clf_unigram, best_clf_bigram


############################################# Random Forest #######################################################
rf_name = 'random forest'
rf = RandomForestClassifier(random_state=37, max_features='sqrt')
param_grid_rf = {f'{rf_name}__n_estimators': [100, 150, 200, 250, 300, 400], f'{rf_name}__max_depth': [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

rf_unigram, rf_bigram = make_predictions(rf, param_grid_rf, rf_name)

##################################################### Logstic Regression ####################################################
lm_name = 'logistic regression'
lm = LogisticRegression(penalty='l1', solver='liblinear', random_state=37)
param_grid_lm = {f'{lm_name}__C': np.logspace(-4, 4, 20), f'{lm_name}__max_iter': [100, 1000, 2500, 5000]}

lm_unigram, lm_bigram = make_predictions(lm, param_grid_lm, lm_name)

###################################################### Decision Tree ###########################################################
dt_name = 'decision tree'
dt = DecisionTreeClassifier(random_state=37)
param_grid_dt = {f'{dt_name}__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], f'{dt_name}__min_impurity_decrease': [
    0.0, 0.001, 0.005, 0.01, 0.015, 0.02], f'{dt_name}__max_features': ['auto', 'sqrt']}

dt_unigram, dt_bigram = make_predictions(dt, param_grid_dt, dt_name)

######################################################### Naive Bayes #############################################
mb_name = 'naive bayes'
mb = MultinomialNB()
param_grid_mb = {f'{mb_name}__alpha': [0.5, 1.0]}
mb_unigram, mb_bigram = make_predictions(mb, param_grid_mb, mb_name)