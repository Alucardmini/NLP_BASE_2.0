#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
from relationship_category.src.attention import attention


class AttLSTM:

    def __init__(self, sentence_length, vocab_size, embedding_size, hidden_size, num_classes, l2_reg_lambda=0.0):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, sentence_length], name='input_x')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None,num_classes], name='input_y')
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
