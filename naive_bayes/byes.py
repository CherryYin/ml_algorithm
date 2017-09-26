#coding:utf-8

import numpy as np

def loadDataSet():
    postingList = [[ 'my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'hime', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'hime'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

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

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    l = len(postingList)
    trainMatrix = []
    for i in range(l):
        trainMatrix.append(setOfWords2Vec(vocabList, postingList[i]))

    p0Vect, p1Vect, pAb = trainNB0(trainMatrix, classVec)
    print p0Vect
    print p1Vect
    print pAb

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0Vect, p1Vect, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0Vect, p1Vect, pAb)

testingNB()







