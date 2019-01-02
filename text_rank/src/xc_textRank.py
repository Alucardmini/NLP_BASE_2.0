#!/usr/bin/python
#coding:utf-8
import jieba


class TextRank(object):

    def __init__(self):
        pass

    def abstract_keys(self, input_text, window_size=5, top_count=10):
        """

        :param input_text:
        :param window_size:
        :param top_count:
        :return:
        """

        # 1. 预处理, 分词去停用词
        with open('../data/stopwords.txt', 'r')as f:
            stop_word_list = [line.strip() for line in f.readlines()]
        input_word_lists = list(jieba.cut(input_text))
        input_word_lists = [word for word in input_word_lists if word not in stop_word_list]

        # 2. 根据窗口大小建立共现矩阵

        # 3. 计算wordscore

        # 4. 输出top

        pass





if __name__ == "__main__":
    src_content = "程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。我取出了百度百科关于“程序员”的定义作为测试用例，很明显，这段定义的关键字应当是“程序员”并且“程序员”的得分应当最高。"
    print(list(jieba.cut(src_content)))
