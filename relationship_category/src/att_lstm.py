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

        initializer = tf.keras.initializers.glorot_normal
        with tf.device("/cpu:0"), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size]))
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.inputs)
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.inputs)

        with tf.variable_scope("dropout-embeddings"):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.rnn_dropout_keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                                                                    cell_fw=fw_cell,
                                                                    cell_bw=bw_cell,
                                                                    inputs=self.embedded_chars,
                                                                    sequence_length=self._length(self.inputs),
                                                                    dtype=tf.float32
                                                                )
        with tf.variable_scope("attention"):
            self.attn, self.alphas = attention(self.rnn_outputs)

        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length