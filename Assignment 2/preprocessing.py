# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:56:14 2022

@author: scro0
"""

import os
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords

############### SET WORKING DIRECTORY TO 'op_spam_v1.4' #############################
stopwords = stopwords.words('english')
    
#Get corpus for training set 
def remove_values(the_list, val):
   return [value for value in the_list if value != val]


corpus = []
labels = [] #1 for deceptive, 0 for truthful
#first loop gets the deceptive and truthful in turn
train_directory = os.path.join(os.getcwd(), 'negative_polarity')
for directory in os.listdir(train_directory):
    #make sure labels are at the same index
    if directory == 'deceptive_from_MTurk':
        label = 1
    else:
        label = 0
    #second loop gets each fold in turn
    e = os.path.join(train_directory, directory)
    for temp in os.listdir(e):
        fold = os.path.join(e, temp)
        #third loop gets each document
        for file in os.listdir(fold):
            tmp = os.path.join(directory, fold, file)
            text = open(tmp).read().split() #split to apply stemming
            for w in stopwords:
                text = remove_values(text, w)
            doc = ''
            #fourth loop stems each word in the document
            for i in range(len(text)):
                word = PorterStemmer().stem(text[i]) #apply stemming, removes about 600 features!
                doc += f' {word}' #convert list back to single string to make compatible with TfidfVectorizer
            corpus.append(doc)
            labels.append(label)
      
#Get corpus for test set            
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
        text = open(tmp).read().split() #split to apply stemming
        
        for w in stopwords:
            text = remove_values(text, w)
            
        doc = ''
        #fourth loop stems each word in the document
        
        for i in range(len(text)):
            word = PorterStemmer().stem(text[i]) #apply stemming, removes about 600 features!
            doc += f' {word}' #convert list back to single string to make compatible with TfidfVectorizer
            
        test_set.append(doc)
        test_labels.append(test_label)
        
#UPDATE: only remove stopwords in CountVectorizer, add 10 featurs to final dataframe
    

pd.DataFrame(corpus).to_csv('train_set.csv', sep=',')
pd.DataFrame(labels).to_csv('train_labels.csv', sep =',')
pd.DataFrame(test_set).to_csv('test_set.csv', sep =',')
pd.DataFrame(test_labels).to_csv('test_labels.csv', sep =',')

'''

def find(s, chars):
    return [i for i, ltr in enumerate(s) for ch in chars if ltr == ch]

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


'''
Question:
    - I guess only tune hyperparameters of the model and not the vectorizer? Otherwise will take too long.
    - What is the feature selection thing with naive Bayes
    -  Should we do bigrams separately? e.g. remove sparse terms differently?
    - Did we remove too many features? Or too little? Or is the bad performance simply due to not
    doing cross validation correctly?
    - How detailed do we have to document everything in the report? E.g. do we need to say how many features
    the data had before and after feature extraction? This is difficult using the combination of pipeline and gridsearch
    as i cannot seem to find a way to retrieve the countvectorizer from the pipeline
'''


'''
We have applied:
    - stemming
    - tokenisation (?)
    - tfidf metric
    - stopwords/punctuation
    - lowercasing
    - Remove sparse terms


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
