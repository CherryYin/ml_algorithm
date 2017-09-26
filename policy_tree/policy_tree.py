#coding:utf-8

import sys
#from tree import *

reload(sys)
sys.setdefaultencoding('utf-8')
from pylab import *


def createDataSet():
    """
创建数据集

    :return:总数据集、特征标签列表、特征值map、类别值列表
    """
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ]
    labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    label_value_map = {u'年龄': [u'青年', u'中年', u'老年'],
                       u'有工作': [u'是', u'否'],
                       u'有房子': [u'是', u'否'],
                       u'信贷情况': [u'一般', u'好', u'非常好']
                       }
    class_value_list = [u'同意', u'拒绝']

    # 返回数据集和每个维度的名称
    return dataSet, labels, label_value_map, class_value_list


"""
H(p) = -∑ (pi * logpi)
计算各分类(y)的经验熵
    feature_class_map: 按类别存储的特征map
    sample_num:样本总数
    base:对数的底，书上对二值分类用2为底，其他是e为底。缺省为2
    return:经验熵
"""
def compute_empirical_entropy(class_map, sample_num, base=2):
    entropy = 0
    for class_x in class_map:
        prob_x = float(class_map[class_x]) / sample_num
        entropy -= prob_x * math.log(prob_x, base)
    return entropy

"""
计算基本的经验熵，H(D)
    dataSet:总体数据集
    class_list:类别列表
    return:基本的经验熵
"""
def compute_base_entroy(dataSet, class_list):
    feature_class_map = {}
    for class_i in class_list:
        feature_class_map[class_i] = 0

    sample_num = 0
    for features in dataSet:
        class_x = features[-1]
        feature_class_map[class_x] += 1
        sample_num += 1
    base_entropy = compute_empirical_entropy(feature_class_map, sample_num)
    return base_entropy

"""
计算条件熵
H(Y|X) = ∑ (pi * H(Y|X=xi)
    dataSet:总的数据集
    i:给定的特征列
    labels:特征维度标签
    label_value_map:特征值map，存储了各特征标签下的值
    return: 条件熵
"""
def comput_condition_entropy(dataSet, i, value_list):
    condition_entropy = 0
    for value in value_list:
        value_num = 0
        sample_num = len(dataSet)
        subdataSet_map = {}
        for j in range(sample_num):
            if value == dataSet[j, i]:
                value_num += 1
                class_x = dataSet[j, -1]
                if class_x not in subdataSet_map.keys():
                    subdataSet_map[class_x] = 1
                else:
                    subdataSet_map[class_x] += 1
        prob_value = float(value_num) / sample_num
        entropy = compute_empirical_entropy(subdataSet_map, value_num)
        condition_entropy += prob_value * entropy

    return condition_entropy

"""
计算信息增益
g(D,A) = H(D) - H(D|A)
    base_entropy:数据集的经验熵
    dataSet:总的数据集
    i：给定的特征列
    value_list:给定特征的特征值列表
    return: 信息增益
"""
def compute_info_gain(base_entropy, dataSet, i, value_list):
    #value_list = label_value_map[labels[i]]
    condition_entropy = comput_condition_entropy(dataSet, i,value_list)
    info_gain = base_entropy - condition_entropy
    return info_gain

"""
计算信息增益比
            g(D,A)
gR(D,A) = -----------
            HA(D)
    info_gain:信息增益
    dataSet:总的数据集
    i：给定的特征列

    return: 信息增益比
"""
def compute_info_gain_rato(info_fain, dataSet, i):
    value_map = {}
    sample_num = len(dataSet)
    entropy_base_i = 0
    for value in dataSet[:, i]:
        if value not in value_map.keys():
            value_map[value] = 1
        value_map[value] += 1
    info_gain_rato = compute_empirical_entropy(value_map, sample_num)

    return info_gain_rato



"""**********************************以下代码基本上是参考别人的**************************************"""
def splitDataSet(dataSet, axis, value):
    """
按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征的维度
    :param value: 特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 删掉这一维特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def chooseBestFeatureToSplitByID3(dataSet, labels, label_value_map, class_list):
    """
选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    base_entropy = compute_base_entroy(dataSet, class_list)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        print i
        value_list = label_value_map[labels[i]]
        infoGain = compute_info_gain(base_entropy, dataSet, i, value_list)
        if (infoGain > bestInfoGain):  # 选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度


"""
返回出现次数最多的分类名称
    :param classList: 类列表
    :return: 出现次数最多的类名称
    """
def majorityCnt(classList):
    classCount = {}  # 这是一个字典
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
创建决策树
    :param dataSet:数据集
    :param labels:数据集每一维的名称
    :return:决策树
    """
def createTree(dataSet, labels, label_value_map, calss_value_list, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    classList = [example[-1] for example in dataSet]  # 类别列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 当类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:  # 当只有一个特征的时候，遍历完所有实例返回出现次数最多的类别
        return majorityCnt(classList)
    dataSet_array = np.array(dataSet)
    bestFeat = chooseBestFeatureToSplitFunc(dataSet_array, labels, label_value_map, class_value_list)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, label_value_map, class_value_list)
    return myTree


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
##################################

# 测试决策树的构建
myDat, labels,label_value_map, class_value_list  = createDataSet()
#myDat = np.array(myDat)
myTree = createTree(myDat, labels, label_value_map, class_value_list)
# 绘制决策树
from policy_tree import treeplotter

treeplotter.createPlot(myTree)