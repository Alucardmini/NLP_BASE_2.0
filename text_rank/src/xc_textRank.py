#!/usr/bin/python
#coding:utf-8
import jieba
import math


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
        # word_2_indces = {input_word_lists[i]: i for i in range(len(input_word_lists))}
        # indeces_2_words = {v: k for k, v in word_2_indces.items()}
        nodes = list(set(input_word_lists))  # 去重复后的word为顶点, 暂时不考虑重复, 重复出现的单词直接刷新窗口,将词添加到第一个出现的词中

        # 2. 计算wordscore
        word_scores = self.calc_word_score(input_word_lists, window_size)

        # 3. 输出top
        word_scores_keys = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
        word_scores_keys = word_scores_keys[:top_count]
        return word_scores_keys

    def calc_word_score(self, input_word_lists, window_size):
        word_scores, cocurrent_map = {}, {}
        init_prob = 1.0 / len(input_word_lists)
        for index, word in enumerate(input_word_lists):
            if word not in word_scores:
                word_scores[word] = init_prob
                cocurrent_map[word] = set(
                    input_word_lists[max(0, index - window_size): min(len(input_word_lists), index + window_size)])
            else:
                word_scores[word] += init_prob
                cocurrent_map[word] |= set(input_word_lists[
                                           max(0, index - window_size): min(len(input_word_lists),
                                                                            index + window_size)])
            tmp_set = set(cocurrent_map[word])

            if word in tmp_set:
                cocurrent_map[word].remove(word)


        # test_nodes = ['A', 'B', 'C', 'D']
        # word_scores = {word: 1/len(test_nodes) for word in test_nodes}
        # cocurrent_map={'A': set(),
        #                'B': set(['A', 'D']),
        #                'C': set('B'),
        #                'D': set(['C', 'A'])}

        d = 0.85  # 阻尼系数
        # 假设32步以后概率转移就OK了吧

        for step in range(32):
            tmp_word_scores = {}
            for word, prob in word_scores.items():
                new_score = math.fabs(1.0 - d)
                sumj = 0.0
                if word in cocurrent_map:
                    for wj in cocurrent_map[word]:
                        # 如果有出度
                        if len(cocurrent_map[wj]) > 0:
                            sumj +=(word_scores[wj] / len(cocurrent_map[wj]))
                        else:
                            # 如果无出度
                            sumj += word_scores[wj]
                new_score += d * sumj
                tmp_word_scores[word] = new_score
            word_scores = tmp_word_scores

        return word_scores




if __name__ == "__main__":
    src_content = "程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。我取出了百度百科关于“程序员”的定义作为测试用例，很明显，这段定义的关键字应当是“程序员”并且“程序员”的得分应当最高。"
    # print(list(jieba.cut(src_content)))
    textrank = TextRank()
    print(textrank.abstract_keys(src_content))
    # tmp1 = set([1, 2, 3])
    # tmp2 = [2, 3, 5]
    # tmp1 |= set(tmp2)
    # print(tmp1)
    # tmp1.remove(2)
    # print(tmp1)



