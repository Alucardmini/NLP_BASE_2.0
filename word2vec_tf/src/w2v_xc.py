#!/usr/bin/python
#coding:utf-8
import jieba
import tensorflow as tf
import collections
import numpy as np
from tqdm import tqdm
import math

class Word2Vec(object):

    def __init__(self, wordlist=None,
                 window_size=3,
                 mode='cbow',
                 embeding_size=200,
                 lr_rate=0.1,
                 batch_size=None,
                 model_path=None,
                 num_sampled=1000,
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
            self.num_sampled    = num_sampled

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
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0)
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 将输入序列向量化
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs)  # batch_size

            # 得到NCE损失
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.train_labels,
                    inputs=embed,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocab_size
                )
            )
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rate).minimize(loss)
            # 变量初始化
            self.init = tf.global_variables_initializer()

            self.merged_summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver()

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)


    def train_by_sentence(self, input_sentence=[]):
        # input_sentences = [sent1, sent2, sent3, ..., sentN]
        # sent 是词序列 ["我", "在", "北京"]
        # cbow 模型
        inputs, labels = [], []
        for sent in input_sentence:
            for index in range(len(sent)):
                start, end = max(0, index - self.window_size), min(sent.__len__(), index + self.window_size)
                for i in range(start, end):
                    if i == index:
                        continue
                    inputs.append(sent[i])
                    labels.append(sent[index])
            if len(labels) == 0:
                continue
        inputs = [self.word2id[x] for x in inputs]
        labels = [self.word2id[x] for x in labels]

        inputs = np.array(inputs, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        labels = np.reshape(labels, [labels.__len__(), 1])

        feed_dict = {
            self.train_inputs: inputs,
            self.train_labels: labels
        }
        _, loss_val, summary_str = self.sess.run([self.train_op, self.loss, self.merged_summary_op], feed_dict=feed_dict)

        self.train_loss_records.append(loss_val)
        self.train_loss_k10 = np.mean(self.train_loss_records)
        if self.train_sent_nums %1000 == 0:
            self.summary_writer.add_summary(summary_str, self.train_sent_nums)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sent_nums, b=self.train_loss_k10))
        # train times
        self.train_word_nums += inputs.__len__()
        self.train_sent_nums += input_sentence.__len__()
        self.train_epochs += 1


if __name__ == "__main__":
    stop_word_path = '../data/stop_words.txt'
    input_path = '../data/280.txt'

    print("loading data_sets ... ... ")
    with open(stop_word_path, 'r')as f:
        lines = f.readlines()
        stop_list = [x.strip() for x in lines]
    words, sentence_list = [], []
    # with open(input_path, 'r', encoding='gbk')as f:
    with open(input_path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
        lines = [line.replace(' ', '').replace('\n', '') for line in lines]
        for i in tqdm(range(len(lines))):
            cut_sents = [x for x in jieba.cut(lines[i]) if x not in stop_list and x not in ['qingkan520','www','com','http', '\u3000']]
            words.extend(cut_sents)
            sentence_list.append(cut_sents)
        print('length of raw words {0}'.format(len(words)))
        words = list(set(words))
        print('length of unique words {0}'.format(len(words)))

        w2v = Word2Vec(wordlist=words, window_size=3, embeding_size=128, num_sampled=10)
        num_steps = 100000
        for i in range(num_steps):
            sent = sentence_list[i % len(sentence_list)]
            w2v.train_by_sentence([sent])

