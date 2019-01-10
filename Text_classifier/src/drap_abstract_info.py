#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@software: PyCharm Community Edition
@file: drap_abstract_info.py
@time: 1/8/19 10:34 AM
"""
import sys
import os
import codecs

from textrank4zh import TextRank4Keyword, TextRank4Sentence

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))[0])

label_list = ["工程类别"]
data_path = rootPath + '/data'
folder = os.path.join(data_path, label_list[0])
file_names = os.listdir(folder)

# tr4w = TextRank4Keyword()

tr4s = TextRank4Sentence()
for name in file_names[0:10]:
    with open(os.path.join(folder, name), 'r')as f:
        text = f.read()
        tr4s.analyze(text=text, lower=True, source='all_filters')

        for item in tr4s.get_key_sentences(num=5):
            print(name, item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
        print('\n')

