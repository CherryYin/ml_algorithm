#! /usr/bin/env python
# coding=utf-8

# Authors: Hanxiaoyang <hanxiaoyang.ml@gmail.com>
# simple naive bayes classifier to classify sohu news topic
# data can be downloaded in http://www.sogou.com/labs/dl/cs.html

# 代码功能：简易朴素贝叶斯分类器，用于对搜狐新闻主题分类，数据可在http://www.sogou.com/labs/dl/cs.html下载(精简版)
# 详细说明参见博客http://blog.csdn.net/han_xiaoyang/article/details/50629608
# 作者：寒小阳<hanxiaoyang.ml@gmail.com>

import os,sys, math, random, collections


def shuffle(inFile):
    '''
        简单的乱序操作，用于生成训练集和测试集
    '''
    textLines = [line.strip() for line in open(inFile)]
    print "正在准备训练和测试数据，请稍后..."
    random.shuffle(textLines)
    num = len(textLines)
    trainText = textLines[:3 * num / 5]
    testText = textLines[3 * num / 5:]
    print "准备训练和测试数据准备完毕，下一步..."
    return trainText, testText


# 总共有9种新闻类别，我们给每个类别一个编号
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


def label2id(label):
    for i in xrange(len(labels)):
        if label == labels[i]:
            return i
    raise Exception('Error lable %s' % (label))


def doc_dict():
    '''
        构造和类别数等长的0向量
    '''
    return [0] * len(labels)


def mutual_info(N, Nij, Ni_, N_j):
    '''
        计算互信息，这里log的底取为2
    '''
    return Nij * 1.0 / N * math.log(N * (Nij + 1) * 1.0 / (Ni_ * N_j)) / math.log(2)


def count_for_cates(trainText, featureFile):
    '''
        遍历文件，统计每个词在每个类别出现的次数，和每类的文档数
        并写入结果特征文件
    '''
    docCount = [0] * len(labels)
    wordCount = dict()   #collections.defaultdict(list)
    # 扫描文件和计数
    for line in trainText:
        label, text = line.strip().split(' ', 1)
        index = label2id(label[0])
        words = text.strip().split(' ')
        for word in words:
            word = word.strip()
            if word == '':continue
            if word not in wordCount:
                wordCount[word] = doc_dict()
            wordCount[word][index] += 1
            docCount[index] += 1
    # 计算互信息值
    print "计算互信息，提取关键/特征词中，请稍后..."
    miDict = dict()   #collections.defaultdict(list)
    N = sum(docCount)
    for k, vs in wordCount.items():
        miDict[k] = doc_dict()
        for i in xrange(len(vs)):
            N11 = vs[i]
            N10 = sum(vs) - N11
            N01 = docCount[i] - N11
            N00 = N - N11 - N10 - N01
            mi = mutual_info(N, N11, N10 + N11, N01 + N11) + \
                 mutual_info(N, N10, N10 + N11, N00 + N10) + \
                 mutual_info(N, N01, N01 + N11, N01 + N00) + \
                 mutual_info(N, N00, N00 + N10, N00 + N01)
            miDict[k][i] = mi
    fWords = set()
    for i in xrange(len(docCount)):
        keyf = lambda x: x[1][i]
        sortedDict = sorted(miDict.items(), key=keyf, reverse=True)
        for j in xrange(250):
            fWords.add(sortedDict[j][0])
    out = open(featureFile, 'w')
    # 输出各个类的文档数目
    out.write(str(docCount) + "\n")
    # 输出互信息最高的词作为特征词
    for fword in fWords:
        out.write(fword + "\n")
    print "特征词写入完毕..."
    out.close()


def load_feature_words(featureFile):
    '''
        从特征文件导入特征词
    '''
    f = open(featureFile)
    # 各个类的文档数目
    docCounts = eval(f.readline())
    features = set()
    # 读取特征词
    for line in f:
        features.add(line.strip())
    f.close()
    return docCounts, features


def train_bayes(featureFile, trainText, modelFile):
    '''
        训练贝叶斯模型，实际上计算每个类中特征词的出现次数
    '''
    print "使用朴素贝叶斯训练中..."
    docCounts, features = load_feature_words(featureFile)
    wordCount = dict()    #collections.defaultdict(list)
    for word in features:
        wordCount[word] = doc_dict()
    # 每类文档特征词出现的次数
    tCount = [0] * len(docCounts)
    for line in open(trainText):
        lable, text = line.strip().split(' ', 1)
        index = label2id(lable[0])
        words = text.split(' ')
        for word in words:
            if word in features:
                tCount[index] += 1
                wordCount[word][index] += 1
    outModel = open(modelFile, 'w')
    # 拉普拉斯平滑
    print "训练完毕，写入模型..."
    for k, v in wordCount.items():
        scores = [(v[i] + 1) * 1.0 / (tCount[i] + len(wordCount)) for i in xrange(len(v))]
        outModel.write(k + "\t" + str(scores) + "\n")
    outModel.close()


def load_model(modelFile):
    '''
        从模型文件中导入计算好的贝叶斯模型
    '''
    print "加载模型中..."
    f = open(modelFile)
    scores = {}
    for line in f:
        word,counts = line.strip().rsplit('\t',1)
        scores[word] = eval(counts)
    f.close()
    return scores


def predict(featureFile, modelFile, testText):
    '''
        预测文档的类标，标准输入每一行为一个文档
    '''
    docCounts, features = load_feature_words(featureFile)
    docScores = [math.log(count * 1.0 / sum(docCounts)) for count in docCounts]
    scores = load_model(modelFile)
    rCount = 0
    docCount = 0
    print "正在使用测试数据验证模型效果..."
    for line in testText:
        lable, text = line.strip().split(' ', 1)
        index = label2id(lable[0])
        words = text.split(' ')
        preValues = list(docScores)
        for word in words:
            if word in features:
                for i in xrange(len(preValues)):
                    preValues[i] += math.log(scores[word][i])
        m = max(preValues)
        pIndex = preValues.index(m)
        if pIndex == index:
            rCount += 1
        # print lable,lables[pIndex],text
        docCount += 1
    print("总共测试文本量: %d , 预测正确的类别量: %d, 朴素贝叶斯分类器准确度:%f" % (rCount, docCount, rCount * 1.0 / docCount))

if __name__=="__main__":
    """
    if len(sys.argv) != 4:
        print "Usage: python naive_bayes_text_classifier.py sougou_news.txt feature_file.out model_file.out"
        sys.exit()

    inFile = sys.argv[1]
    featureFile = sys.argv[2]
    modelFile = sys.argv[3]
    """
    abs_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
    root_path = os.path.join(abs_path, "data\\SogouC.reduced\\")
    inFile = os.path.join(root_path, 'splited_text.txt')
    featureFile = os.path.join(root_path, 'feature.txt')
    modelFile = os.path.join(abs_path, 'model/nb_model')
    trainText, testText = shuffle(inFile)
    count_for_cates(trainText, featureFile)
    train_bayes(featureFile, inFile, modelFile)
    predict(featureFile, modelFile, testText)