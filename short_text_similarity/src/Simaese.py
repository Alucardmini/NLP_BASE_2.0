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
        line = line.strip()
        index, left, right, label = line.split('\t')
        left_data.append(left)
        right_data.append(right)
        labels.append(int(label))

custom_words = ['花呗', '支付宝', '借呗', '闲鱼', '蚂蚁借呗', '蚂蚁花呗', '封号', '摇一摇']
for _ in custom_words:
    jieba.add_word(_)

left_data = [[x for x in list(jieba.cut(sent)) if x not in stop_word_list] for sent in left_data]
right_data = [[x for x in list(jieba.cut(sent)) if x not in stop_word_list] for sent in right_data]

word_2_indices = {'<UNK>': 0}
if len(left_data) != len(right_data):
    raise("data length unequal")

for i in range(len(left_data)):
    for word in left_data[i]:
        if word not in word_2_indices:
            word_2_indices[word] = len(word_2_indices)
    for word in right_data[i]:
        if word not in word_2_indices:
            word_2_indices[word] = len(word_2_indices)

indices_2_word = {index: word for index, word in word_2_indices.items()}

def vector_sent(sent, word_2_indices):
    return [word_2_indices[word] for word in sent if word in word_2_indices]

#相似度计算
def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


left_data = [vector_sent(sent, word_2_indices) for sent in left_data]
right_data = [vector_sent(sent, word_2_indices) for sent in right_data]

embedding_size = 256
max_length = 30
n_hidden = 50
gradient_clipping_norm = 1.25
n_epoch = 15
batch_size = 128

train_size = int(len(left_data) * 0.8)

left_data = pad_sequences(left_data, maxlen=max_length)
right_data = pad_sequences(right_data, maxlen=max_length)

train_left = left_data[:train_size]
train_right = right_data[:train_size]
train_label = labels[:train_size]

test_left = left_data[train_size:]
test_right = right_data[train_size:]
test_label = labels[train_size:]

left_input = Input(shape=(max_length,), dtype='int32')
right_input = Input(shape=(max_length,), dtype='int32')

embedding_layer = Embedding(len(left_data), embedding_size)
left_embeding = embedding_layer(left_input)
right_embeding = embedding_layer(right_input)

shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(left_embeding)
right_output = shared_lstm(right_embeding)

import keras
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# malstm_distance = keras.layers.merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
#                         output_shape=lambda x: (x[0][0], 1))([left_output, right_output])


malstm = Model([left_input, right_input], [malstm_distance])
optimizer = Adadelta(clipnorm=gradient_clipping_norm)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
malstm.fit(x=[np.asarray(train_left), np.asarray(train_right)], y=train_label, batch_size=batch_size, epochs=n_epoch,
                           validation_data=([np.asarray(test_left), np.asarray(test_right)], test_label))
