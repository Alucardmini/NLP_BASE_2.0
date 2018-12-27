#!/usr/bin/python
#coding:utf-8
import pickle
import os
import operator
from functools import reduce


class ner_datas(object):

    def __init__(self):
        self.pos_dict = {'B-ORG': 1,
                         'I-ORG': 2,
                         'B-PER': 3,
                         'I-PER': 4,
                         'B-LOC': 5,
                         'I-LOC': 6,
                         'O': 0
                         }
        self.unk = '<UNK>'

    def read_data_from_path(self, input_path):
        vocabs, labels = [], []
        with open(input_path, 'r') as f:
            tmp_vocab, tmp_label = [], []
            for line in f.readlines():
                content_list = line.strip().split('\t')
                if len(content_list) == 2:
                    tmp_vocab.append(content_list[0])
                    tmp_label.append(content_list[1])
                else:
                    vocabs.append(tmp_vocab)
                    labels.append(tmp_label)
                    tmp_vocab, tmp_label = [], []

        return vocabs, labels

    def vector_sent_only(self, sent):
        vector_input = []
        if not hasattr(self, 'word2id'):
            word2id_path = '../data/word2id.pkl'
            if os.path.exists(word2id_path):
                with open(word2id_path, 'rb')as f:
                    self.word2id = pickle.load(f)
        for i, input in enumerate(sent):
            if input in self.word2id:
                vector_input.append(self.word2id[input])
            else:
                vector_input.append(0)
        return vector_input



    def vector_sent(self, sent, label):
        vector_input, vector_label = [], []

        if not hasattr(self, 'word2id'):
            word2id_path = '../data/word2id.pkl'
            if os.path.exists(word2id_path):
                with open(word2id_path, 'rb')as f:
                    self.word2id = pickle.load(f)

        for i, (input, label) in enumerate(zip(sent, label)):
            if input in self.word2id and label in self.pos_dict:
                vector_input.append(self.word2id[input])
                vector_label.append(self.pos_dict[label])
            else:
                vector_input.append(0)
                vector_label.append(self.pos_dict['O'])
        return vector_input, vector_label

    def vector_data(self, inputs, labels):
        vector_input, vector_labels = [], []
        for i, (input, label) in enumerate(zip(inputs, labels)):
            vector_sent, vector_label = self.vector_sent(input, label)
            vector_input.append(vector_sent)
            vector_labels.append(vector_label)

        return vector_input, vector_labels

    def load_data(self):
        train_data_path = '../data/train_data'
        test_data_path = '../data/test_data'

        word2id_path = '../data/word2id.pkl'

        train_vocabs, train_labels = self.read_data_from_path(train_data_path)
        test_vocabs, test_labels = self.read_data_from_path(test_data_path)

        if os.path.exists(word2id_path):
            with open(word2id_path, 'rb')as f:
                self.word2id = pickle.load(f)
        else:

            unique_vocab = set()

            for vocab in train_vocabs:
                unique_vocab |= set(vocab)
            unique_vocab = list(unique_vocab)
            self.word2id = {unique_vocab[i]: i+1 for i in range(len(unique_vocab))}
            self.word2id[self.unk] = 0
            with open(word2id_path, 'wb')as f:
                pickle.dump(self.word2id, f)
        train_vocabs, train_labels = self.vector_data(train_vocabs, train_labels)
        test_vocabs, test_labels = self.vector_data(test_vocabs, test_labels)

        return (train_vocabs, train_labels), (test_vocabs, test_labels)

if __name__ == '__main__':

    data_loader = ner_datas()
    # (train_vocabs, test_labels), (test_vocabs, test_labels) = data_loader.load_data()
    # print(train_vocabs.__len__())
    test_content = "北京到上海多少公里"
    pred_x = data_loader.vector_sent_only(list(test_content))
    print(pred_x)