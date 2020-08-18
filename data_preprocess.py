from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import pickle

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Read data from files
train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv("data/testData.tsv", header=0,
                   delimiter="\t", quoting=3)

# 获取列名
print([column for column in train])
print(train.head())
print(train.shape)
print(train.iloc[0])
print(train['review'][0])
print(BeautifulSoup(train['review'][0], "lxml").get_text())
print(train['review'][0])
print(BeautifulSoup(train['review'][0], "lxml").get_text())
print(len(train))

print("len str")
tmp = BeautifulSoup(train['review'][0], "lxml").get_text()
print(tmp)
print(tmp.strip('"'))
tmp = BeautifulSoup(train['review'][1], "lxml").get_text()


tmp = tmp.strip('"')
print(tmp)
tmp = tmp.replace('\\', '')
print(tmp)


train = train.drop(['id'], axis=1)
print(train.head())
print(train.shape)
print(train.iloc[1])
#swap 2 columns

cols = list(train)
print(cols[0])
print(cols[1])
print(cols)
cols.insert(0, cols.pop(cols.index('review')))
print(cols)
train = train.loc[:, cols]
print(train.head())



if __name__ == '__main__':

    for i in range(len(train['review'])):
        tmp = BeautifulSoup(train['review'][i], "lxml").get_text()
        tmp = tmp.strip('"')
        tmp = tmp.replace('\\', '')
        train['review'][i] = tmp
    train.columns = ['sentence', 'label']
    print(train.head())

    X_train, X_dev, Y_train, Y_dev = train_test_split(train['sentence'], train['label'], test_size=0.25, stratify=train['label'], random_state=1,
                                                      #shuffle=False
                                                      )
    train_print = pd.DataFrame({'sentence' : X_train,
                                'label' : Y_train})
    dev_print = pd.DataFrame({'sentence': X_dev,
                             'label': Y_dev})

    train_print = train_print.sort_index()
    dev_print = dev_print.sort_index()
    print(train_print.head())
    print(dev_print.head())


    for i in range(len(test['review'])):
        test['id'][i] = i
        tmp = BeautifulSoup(test['review'][i], "lxml").get_text()
        tmp = tmp.strip('"')
        tmp = tmp.replace('\\', '')
        test['review'][i] = tmp
    test.columns = ['index', 'sentence']

    print(test.head())
    train_print.to_csv('data/train.tsv', index=False, sep='\t')
    dev_print.to_csv('data/dev.tsv', index=False, sep='\t')
    test.to_csv('data/test.tsv', index=False, sep='\t')