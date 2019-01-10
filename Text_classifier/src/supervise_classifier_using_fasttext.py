# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '1/6/19'

# -*- coding:utf-8 -*-
import pandas as pd
import random
import fasttext
import jieba
from sklearn.model_selection import train_test_split

cate_dic = {'工程类别': 1, '电梯': 2, '监理': 3, '设计': 4}

"""
函数说明：加载数据
"""
def loadData():

    names = 'label_index, custom_label, content'.split(',')
    #利用pandas把数据读进来
    df_technology = pd.read_csv("../train_data/工程类别.csv",encoding ="utf-8", names=names )
    df_technology=df_technology.dropna()    #去空行处理

    df_car = pd.read_csv("../train_data/电梯.csv",encoding ="utf-8", names=names)
    df_car=df_car.dropna()

    df_entertainment = pd.read_csv("../train_data/监理.csv",encoding ="utf-8", names=names)
    df_entertainment=df_entertainment.dropna()

    df_military = pd.read_csv("../train_data/设计.csv",encoding ="utf-8", names=names)
    df_military=df_military.dropna()

    names = df_technology.columns.values.tolist()
    technology=df_technology[names[-1]].values.tolist()
    car=df_car[names[-1]].values.tolist()
    entertainment=df_entertainment[names[-1]].values.tolist()
    military=df_military[names[-1]].values.tolist()

    return technology,car,entertainment,military


"""
函数说明：停用词
参数说明：
    datapath：停用词路径
返回值：
    stopwords:停用词
"""
def getStopWords(datapath):
    stopwords=pd.read_csv(datapath,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
    stopwords=stopwords["stopword"].values
    return stopwords

"""
函数说明：去停用词
参数：
    content_line：文本数据
    sentences：存储的数据
    category：文本类别
"""
def preprocess_text(content_line,sentences,category,stopwords):
    for line in content_line:
        try:
            segs=jieba.lcut(line)    #利用结巴分词进行中文分词
            segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
            segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
            sentences.append("__label__"+str(category)+" , "+" ".join(segs))    #把当前的文本和对应的类别拼接起来，组合成fasttext的文本格式
        except Exception as e:
            print (line)
            continue

"""
函数说明：把处理好的写入到文件中，备用
参数说明：

"""
def writeData(sentences,fileName):
    print("writing data to fasttext format...")
    out=open(fileName,'w')
    for sentence in sentences:
        out.write(sentence+"\n")
    print("done!")

"""
函数说明：数据处理
"""
def preprocessData(stopwords,saveDataFile):
    technology,car,entertainment,military=loadData()

    #去停用词，生成数据集
    sentences=[]
    preprocess_text(technology,sentences,cate_dic["工程类别"], stopwords)
    preprocess_text(car,sentences,cate_dic["电梯"], stopwords)
    preprocess_text(entertainment,sentences,cate_dic["监理"], stopwords)
    preprocess_text(military,sentences,cate_dic["设计"], stopwords)

    random.shuffle(sentences)    #做乱序处理，使得同类别的样本不至于扎堆

    writeData(sentences,saveDataFile)

if __name__=="__main__":
    # stopwordsFile="../data/stop_words.txt"
    # stopwords=getStopWords(stopwordsFile)
    saveDataFile=r'train_data.txt'
    # preprocessData(stopwords,saveDataFile)
    # classifier=fasttext.supervised(saveDataFile, 'classifier.model', label_prefix='__lable__')
    # result = classifier.test(saveDataFile)
    # print("P@1:",result.precision)    #准确率
    # print("R@2:",result.recall)    #召回率
    # print("Number of examples:", result.nexamples)    #预测错的例子
    #
    # #实际预测
    # lable_to_cate={1:'工程类别', 2:'电梯',3:'监理',4:'设计'}
    #
    # # texts=['中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先']
    # texts = ['全国重点文物保护 单位 祁门 戏台 保护 规划 编制 采购 项目 合同 备案 全国 重点 文物保护 单位 祁门 戏台 保护 规划 编制 采购 项目 合同 备案 发布 日期 2018 11 26 15 35 全国 重点 文物保护 单位 祁门 戏台 保护 规划 编制 采购 项目 合同 备案 合同 编号 皖祁 ZC2018G009A 合同 名称 全国 重点 文物保护 单位 祁门 戏台 保护 规划 编制 采购 项目 项目编号 皖祁 ZC2018G009 项目名称 全国 重点 文物保护 单位 祁门 戏台 保护 规划 编制 采购 项目 采购 甲方 祁门县 文化 广电 新闻出版局 供应商 乙方 同济大学 建筑设计 研究院 集团 有限公司 所属 地域 祁门县 所属 行业 合同 金额 790000.0000 合同 签订 日期 合同 公告 日期 2018 11 26 代理 机构 祁门县 公共资源 交易中心 免责 声明 页面 提供 政府 采购 合同 中华人民共和国政府 采购 实施 条例 采购 发布 本网 内容 概不负责 承担 法律责任'.replace(' ', '')]
    #
    # lables=classifier.predict(texts)
    # print(lables)
    # # print(lable_to_cate[int(lables[0][0])])
    #
    # #还可以得到类别+概率
    # lables=classifier.predict_proba(texts)
    # print(lables)
    #
    # #还可以得到前k个类别
    # lables=classifier.predict(texts, k=3)
    # print(lables)
    #
    # #还可以得到前k个类别+概率
    # lables=classifier.predict_proba(texts, k=3)
    # print(lables)
    #
    # texts = ['它被誉为"天下第一果"，补益气血，养阴生津，现在吃正应季!  六七月是桃子大量上市的季节，因其色泽红润，肉质鲜美，有个在实验基地里接受治疗的妹子。广受大众的喜爱。但也许你并不知道，看惯了好莱坞大片眼花缭乱的特效和场景。它的营养也是很高的，不仅富含多种维生素、矿物质及果酸，至少他们一起完成了一部电影，其含铁量亦居水果之冠，被誉为"天下第一果"。1、在来世那个平行世界的自己。增加食欲，养阴生津的作用，可用于大病之后，气血亏虚，面黄肌瘦，Will在海滩上救下了Isla差点溺水的儿子。心悸气短者。2、最近有一部叫做《爱有来世》的科幻电影。桃的含铁量较高，就越容易发现事情的真相。是缺铁性贫血病人的理想辅助食物。3、桃含钾多，含钠少，适合水肿病人食用。4、桃仁有活血化淤，润肠通作用，可用于闭经、跌打损伤等辅助治疗。胃肠功能弱者不宜吃桃、桃仁提取物有抗凝血作用，而Will也好像陷入魔怔一般。并能抑制咳嗽中枢而止咳，扩展"科学来自于人性"的概念。同时能使血压下降，片中融合了很多哲学、宗教的玄妙概念，可用于高血压病人的辅助治疗。6、桃花有消肿、利尿之效，可用于治疗浮肿腹s水，大便干结，小便不利和脚气足肿。一段美好的故事才就此开始。桃子性热，味甘酸，具有补心、解渴、不过都十分注重内核的表达，充饥、生津的功效，父亲没有继续在房间埋头工作']
    # labels = classifier.predict(texts)
    # print(labels)
    # model = fasttext.load_model('classifier.model.bin')
    # print(model.words)
    # print(model['周围环境'])

    train_path = 'train_data.txt'
    # classifier = fasttext.supervised(train_path, 'xc_model', label_prefix='__lable__')
    # result = classifier.test(train_path)
    # print('P@1:', result.precision)
    # print('R@1:', result.recall)
    # print('Number of examples:', result.nexamples)
    test_path = 'test_data.txt'

    classifier = fasttext.load_model("xc_model.bin", label_prefix="__label__")
    result = classifier.test(test_path, k=1)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)
