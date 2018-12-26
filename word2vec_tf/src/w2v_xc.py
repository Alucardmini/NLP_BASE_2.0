#!/usr/bin/python
#coding:utf-8
import jieba
import tensorflow as tf
import collections
import numpy as np
from tqdm import tqdm

class Word2Vec(object):

    def __init__(self, wordlist=None,
                 window_size=3,
                 mode='cbow',
                 embeding_size=200,
                 lr_rate=0.1,
                 batch_size=None,
                 model_path=None,
                 logdir='/tmp/simple_word2vec'):
        self.batch_size = batch_size
        if model_path:
            self.load_model(model_path)
        else:

            assert type(wordlist) == list
            self.wordlist       =   wordlist
            self.vocab_size      =   wordlist.__len__()
            self.window_size    =   window_size
            self.mode           =   mode
            self.embedding_size =   embeding_size
            self.lr_rate        =   lr_rate
            self.logdir         =   logdir
            # construct word2id dict
            self.word2id = {self.wordlist[i]:i for i in range(len(self.wordlist))}

            self.train_word_nums =  0  # 训练的的单词个数
            self.train_sent_nums =  0  # 训练的句子个数
            self.train_epochs    =  0  #　训练的次数
            # train loss records
            self.train_loss_records = collections.deque(maxlen=10) # 保存最近10次的误差
            self.train_loss_k10 = 0
        self.build_graph()
        self.init_op()

    def load_model(self, model_path):
        pass

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.init = tf.global_variables_initializer()
            self.inputs = tf.placeholder(tf.int32, [self.batch_size])
            self.labels = tf.placeholder(tf.int32, [self.batch_size, 1])
            self.embedding_dict = tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)


    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.summary.FileWriter(self.logdir, self.sess.graph)


if __name__ == "__main__":
    stop_word_path = '../data/stop_words.txt'
    input_path = '../data/280.txt'

    print("loading data_sets ... ... ")
    with open(stop_word_path, 'r')as f:
        lines = f.readlines()
        stop_list = [x.strip() for x in lines]
    words = []
    with open(input_path, 'r', encoding='gbk')as f:
        lines = f.readlines()
        lines = [line.replace(' ', '').replace('\n', '') for line in lines]
        for i in tqdm(range(len(lines))):
            words.extend([x for x in jieba.cut(lines[i]) if x not in stop_list and x not in ['qingkan520','www','com','http']])

        print('length of raw words {0}'.format(len(words)))
        words = list(set(words))
        print('length of unique words {0}'.format(len(words)))
