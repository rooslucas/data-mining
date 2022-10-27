# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:56:14 2022

@author: scro0
"""

import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import copy
import pandas as pd
import numpy as np


############### SET WORKING DIRECTORY TO 'op_spam_v1.4' #############################

# Get stop words
stop_words = [',', '.', '?', '!', '"', "'", ";"]
for word in stopwords.words('english'):
    stop_words.append(word)


# Get corpus for training set
def remove_values(the_list, val):
    return [value for value in the_list if value != val]


corpus = []
labels = []  # 1 for deceptive, 0 for truthful
# first loop gets the deceptive and truthful in turn
train_directory = os.path.join(os.getcwd(), 'negative_polarity')
for directory in os.listdir(train_directory):
    # make sure labels are at the same index
    if directory == 'deceptive_from_MTurk':
        label = 1
    else:
        label = 0
    # second loop gets each fold in turn
    e = os.path.join(train_directory, directory)
    for temp in os.listdir(e):
        fold = os.path.join(e, temp)
        # third loop gets each document
        for file in os.listdir(fold):
            tmp = os.path.join(directory, fold, file)
            text = open(tmp).read().split()  # split to apply stemming
            for w in stop_words:
                text = remove_values(text, w)
            doc = ''
            # fourth loop stems each word in the document
            for i in range(len(text)):
                # apply stemming, removes about 600 features!
                word = PorterStemmer().stem(text[i])
                # convert list back to single string to make compatible with TfidfVectorizer
                doc += f' {word}'
            corpus.append(doc)
            labels.append(label)

# Get corpus for test set
test_directory = os.path.join(os.getcwd(), 'test_data')

test_set = []
test_labels = []

for fold in os.listdir(test_directory):
    print(fold)
    if fold == 'fold5_deceptive':
        test_label = 1
    else:
        test_label = 0
    e = os.path.join(test_directory, fold)
    for file in os.listdir(e):
        tmp = os.path.join(e, file)
        text = open(tmp).read().split()  # split to apply stemming
        for w in stop_words:
            text = remove_values(text, w)
        doc = ''
        # fourth loop stems each word in the document
        for i in range(len(text)):
            # apply stemming, removes about 600 features!
            word = PorterStemmer().stem(text[i])
            # convert list back to single string to make compatible with TfidfVectorizer
            doc += f' {word}'
        test_set.append(doc)
        test_labels.append(test_label)


################################################## UNIGRAMS ############################################################


cv1_train = CountVectorizer(stop_words=stop_words,
                            lowercase=True, ngram_range=(1, 1), min_df=0.05)
counts_unigram_train = cv1_train.fit_transform(corpus)
counts_unigram_test = cv1_train.transform(test_set)

# variables containing feature names and the actual values. Can use these for training the model
unigram_features = cv1_train.get_feature_names_out()
unigram_train_values = counts_unigram_train.toarray()
unigram_test_values = counts_unigram_test.toarray()

################################################## BIGRAMS ############################################################


def find(s, chars):
    return [i for i, ltr in enumerate(s) for ch in chars if ltr == ch]


bigram_train = copy.deepcopy(corpus)
bigram_test = copy.deepcopy(test_set)
'''

Can add sentence markers, but there are some things that still go wrong, maybe best to keep it simple and mention in report

def add_sentence_tokens(corpus):
    for i in range(len(corpus)):
        doc = corpus[i]
        idx = find(doc, ['.', '!', '?'])
        new_str = '<s>'
        old_j = 0
        for j in idx:
            new_str = new_str + doc[old_j:(j-1)] + ' <\s>' + '.' + ' <s>'
            old_j = j+1
        corpus[i] = new_str
    return corpus

bigram_train = add_sentence_tokens(bigram_train)
bigram_test = add_sentence_tokens(bigram_test)
        
'''
cv2 = CountVectorizer(stop_words=stop_words, lowercase=True,
                      ngram_range=(1, 2), min_df=0.05)

counts_bigram_train = cv2.fit_transform(bigram_train)
counts_bigram_test = cv2.transform(bigram_test)

# variables containing feature names and the actual values. Can use these for training the model
bigram_features = cv2.get_feature_names_out()
#bigram_features = np.append(bigram_features, 'label')
bigram_train_values = np.concatenate(
    (counts_bigram_train.toarray(), np.reshape(labels, (len(labels), 1))), axis=1)
bigram_test_values = counts_bigram_test.toarray()


def make_df(features, count_matrix, labels):
    features = np.append(features, 'label')
    values = np.concatenate(
        (count_matrix.toarray(), np.reshape(labels, (len(labels), 1))), axis=1)
    df = pd.DataFrame(values, columns=features)
    return df


train_unigram_df = make_df(unigram_features, counts_unigram_train, labels)
test_unigram_df = make_df(unigram_features, counts_unigram_test, test_labels)
train_bigram_df = make_df(bigram_features, counts_bigram_train, labels)
test_bigram_df = make_df(bigram_features, counts_bigram_test, labels)


'''
We have applied:
    - stemming
    - tokenisation (?)
    - tfidf metric
    - stopwords/punctuation
    - lowercasing
    - smoothing

TODO:
    - Remove sparse terms
    - Remove frequent terms (?)
    
Questions:
    - Is it advisable to use tfidf features as multinomial naive Bayes needs
    integer feature counts 
    
    - If we remove sparse terms, the number of features will differ between unigram and bigram features
'''

'''
Code to get original folds back
'''

'''
x_folds = [[], [], [], [], []]
y_folds = [[], [], [], [], []]
indices = range(0,len(labels)+1, 160)
for i in range(5):
    print(i)
    x_folds[i] = corpus[indices[i]: indices[i+1]]
    y_folds[i] = labels[indices[i]: indices[i+1]]
    

'''
