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


class node:
    def __init__(self, point, label):
        self.left = None
        self.right = None
        self.point = point
        self.label = label
        self.parent = None
        pass

    def set_left(self, left):
        if left == None: pass
        left.parent = self
        self.left = left

    def set_right(self, right):
        if right == None: pass
        right.parent = self
        self.right = right


def median(lst):
    m = len(lst) / 2
    return lst[m], m


def build_kdtree(data, d):
    data = sorted(data, key=lambda x: x[d.next()])
    p, m = median(data)
    tree = node(p[:-1], p[-1])

    del data[m]

    if m > 0: tree.set_left(build_kdtree(data[:m], d))
    if len(data) > 1: tree.set_right(build_kdtree(data[m:], d))
    return tree


def distance(a, b):
    diff = a - b
    squaredDiff = diff ** 2
    return np.sum(squaredDiff)


def search_kdtree(tree, d, target, k):
    den = d.next()
    if target[den] < tree.point[den]:
        if tree.left != None:
            return search_kdtree(tree.left, d, target, k)
    else:
        if tree.right != None:
            return search_kdtree(tree.right, d, target, k)

    def update_best(t, best):
        if t == None: return
        label = t.label
        t = t.point
        d = distance(t, target)
        for i in range(k):
            if d < best[i][1]:
                for j in range(0, i):
                    best[j][1] = best[j+1][1]
                    best[j][0] = best[j+1][0]
                    best[j][2] = best[j+1][2]
                best[i][1] = d
                best[i][0] = t
                best[i][2] = label
    best = []
    for i in range(k):
        best.append( [tree.point, 100000.0, 10] )
    while (tree.parent != None):
        update_best(tree.parent.left, best)
        update_best(tree.parent.right, best)
        tree = tree.parent
    return best


def testHandWritingClass():
    ## step 1: load data
    print "step 1: load data..."
    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    l = min(train_x.shape[0], train_y.shape[0])
    rows = train_x.shape[1]
    for i in range(l):
        train_x[i, -1] = train_y[i]

    densim = itertools.cycle(range(0, rows-1))
    ## step 2: training...
    print "step 2: build tree..."
    mnist_tree = build_kdtree(train_x, densim)

    ## step 3: testing
    print "step 3: testing..."
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples
    K = 3
    for i in xrange(test_num):
        best_k = search_kdtree(mnist_tree, densim, test_x[i, :-1], K)
        classCount = {}
        for j in range(K):
            voteLabel = best_k[j][2]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        maxCount = 0
        predict = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                predict = key
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print "完成%d张图片"%(i)
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print "一共运行了%d秒"%((b-a).seconds)

    ## step 4: show the result
    print "step 4: show the result..."
    print 'The classify accuracy is: %.2f%%' % (accuracy * 100)

if __name__ == '__main__':
    testHandWritingClass()


