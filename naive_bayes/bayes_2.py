#coding:utf-8

import numpy as np

#coding:utf-8

import numpy as np
import gzip
import itertools
from datetime import datetime

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print magic, num_images, rows, cols
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        #reshape成二维
        data = data.reshape(num_images, rows, cols)
        #二值化
        data_value_binary = np.minimum(data, 1)
        #按行相加,存到钱28个元素中，按列相加，存入后28个元素中
        #如果分类效果不好，可再计算按对角线相加、行列式等
        data_tidy = np.zeros((num_images, rows + cols + 1), dtype=np.uint32)
        for i in range(num_images):
            data_tidy[i, :rows] = np.sum(data_value_binary[i], axis=1)
            data_tidy[i, rows:(rows+cols)] = (np.sum(data_value_binary[i].transpose(), axis=1))
        return data_tidy


#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def createVocabList(dataSet):
    vocabSet = set( [] )
    for document in dataSet:
        #对vocabSet和document做并集，并赋给vocabSet
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" %word
    return returnVec

def bagOfWords2vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" %word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)    #p(ci)
    #p0Num = np.zeros(numWords)
    p0Num = np.ones(numWords)  #采用对数后，为了防止概率太小，初始化为1
    #p1Num = np.zeros(numWords)
    p1Num = np.ones(numWords)
    #p0Denom = 0.0
    #p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  #p(w)
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = p1Num / p1Denom
    #p0Vect = p0Num / p0Denom
    p1Vect = np.log(p1Num/p1Denom)  #由于概率相乘，会产生下溢出，这里将初始概率改为对数概率
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive
