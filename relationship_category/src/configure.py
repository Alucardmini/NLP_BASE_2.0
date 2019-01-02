#!/usr/bin/python
#coding:utf-8

import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_boolean("clean",               True,               "clean train folder")
flags.DEFINE_string("train_path",           "../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT", "path of train_data")
flags.DEFINE_string("test_path",            "../data/SemEval2010_task8_all_data/SemEval2010_task8_testing/TEST_FILE.txt",         "path of test data")
flags.DEFINE_integer("max_sentence_length", 90,                 "max lenght of input sentence")
flags.DEFINE_float("dev_sample_percent",     0.1,               "percent of the train data used")

# embedding
flags.DEFINE_string("embedding_path",       None,               "path of pre-trained word embeddings")
flags.DEFINE_integer("embedding_dim",       100,                "word2vec dim")
flags.DEFINE_float("emb_dropout_keep_prob", 0.7,                "Dropout keep probability of embedding layer (default: 0.7)")
flags.DEFINE_float("dev_sample_percentage", 0.1,                "dev_sample_percentage (default: 0.1)")

# AttLSTM
flags.DEFINE_integer("hidden_size",         100,                "Dimensionality of RNN hidden (default: 100)")
flags.DEFINE_float("rnn_dropout_keep_prob", 0.7,                "Dropout keep probability of RNN (default: 0.7)")

# Misc
flags.DEFINE_string("desc",                 "",                 "Description for model")
flags.DEFINE_float("dropout_keep_prob",     0.5,                "Dropout keep probability of output layer (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda",         1e-5,               "L2 regularization lambda (default: 1e-5)")

# train parameters
flags.DEFINE_integer("batch_size",          128,                "batch_size")
flags.DEFINE_integer("num_epochs",          32,                 "train epochs")
flags.DEFINE_integer("evaluate_every",      100,                 "train epochs")
flags.DEFINE_integer("num_checkpoints",     5,                  "Number of checkpoints to store (default: 5)")
flags.DEFINE_integer("display_every",       10,                 "Number of iterations to display training information")
flags.DEFINE_float("decay_rate",            0.9,                "Decay rate for learning rate (Default: 0.9)")
flags.DEFINE_float("learning_rate",         1.0,                "Which learning rate to start with (Default: 1.0)")

flags.DEFINE_bool("gpu_allow_growth",       True,               "")
flags.DEFINE_bool("log_device_placement",   False,               "")
flags.DEFINE_bool("allow_soft_placement",   True,               "")

FLAGS = flags.FLAGS