#!/usr/bin/python
#coding:utf-8

import pandas as pd
import os
import numpy as np
import jieba

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta,SGD
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# 1. 预处理 加载停用词
with open('../data/stopwords.txt', 'r')as f:
    stop_word_list = [line.strip() for line in f.readlines()]

left_data, right_data, labels = [], [], []
with open('../data/atec_nlp_sim_train.csv', 'r')as f:
    lines = f.readlines()
    for line in lines:
        index, left, right, label = line.split('\t')
        left_data.append(left)
        right_data.append(right)
        labels.append(label)

custom_words = ['花呗', '支付宝', '借呗', '闲鱼', '蚂蚁借呗', '蚂蚁花呗', '封号', '摇一摇']
for _ in custom_words:
    jieba.add_word(_)

left_data = [[x for x in list(jieba.cut(sent)) if x not in stop_word_list] for sent in left_data]
right_data = [[x for x in list(jieba.cut(sent)) if x not in stop_word_list] for sent in right_data]

embedding_size = 256
