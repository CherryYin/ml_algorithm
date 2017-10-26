#coding:utf-8

import numpy as np
import random
import os
from libsvm.python.svmutil import *

def split_data(input_data_path, input_label_path, input_size, K):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    with open(input_data_path, 'r') as f1:
        x_lines = f1.readlines()
    with open(input_label_path, 'r') as f1:
        y_lines = f1.readlines()
    counter = 0
    test_num = 0
    length = min(len(x_lines), len(y_lines))

    for i in range(length):
        Xis = x_lines[i].split(', ')
        yis = y_lines[i].strip()
        if counter % K == 0:
            test_num = random.randint(0, K)
            counter = 0
        if counter == test_num:
            test_x.append([int(Xis[0][1:]), float(Xis[1]), float(Xis[2][:-2])])
            test_y.append(int(yis))
        else:
            train_x.append([int(Xis[0][1:]), float(Xis[1]), float(Xis[2][:-2])])
            train_y.append(int(yis))
        counter += 1

        """数据归一化"""
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    for i in range(input_size):
        divided_max_value = 1 / max(train_x[:, i])
        train_x[:, i] *= divided_max_value
        test_x[:, i] *= divided_max_value
    return train_x.tolist(), train_y, test_x.tolist(), test_y


abs_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
root_path = os.path.join(abs_path, "data\\drama\\")
train_x, train_y, test_x, test_y = split_data(os.path.join(root_path,"features_for_pca.txt"), os.path.join(root_path,"labels.txt"), 5, 8)
print train_x[:10]

print "训练模型开始..."
prob = svm_problem(train_y,train_x,isKernel=True)
param = svm_parameter('-t 4 -c 4 -b 1')
m = svm_train(prob, param)
# For the format of precomputed kernel, please read LIBSVM README.

# Other utility functions
print "测试模型开始 ... "
model_path = os.path.join(abs_path,'model/svm_drama_5.model')
svm_save_model(model_path, m)
m = svm_load_model(model_path)
p_label, p_acc, p_val = svm_predict(test_y, test_x, m, '-b 1')
ACC, MSE, SCC = evaluations(test_y, p_label)
print ACC, MSE, SCC
