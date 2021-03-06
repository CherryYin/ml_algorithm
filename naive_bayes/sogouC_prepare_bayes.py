#coding:utf-8

import os, sys
import jieba
from datetime import datetime

reload(sys)
sys.setdefaultencoding("utf-8")

stopWordList= ['，'.decode('utf-8'), '的'.decode('utf-8'), '\n', '　', '。'.decode('utf-8'), '、'.decode('utf-8'), '在'.decode('utf-8'),
               '了'.decode('utf-8'), '是'.decode('utf-8'), '“'.decode('utf-8'), '”'.decode('utf-8'), '&', 'nbsp', '和'.decode('utf-8'),
                '：'.decode('utf-8'), '：'.decode('utf-8'), '有'.decode('utf-8'), '也'.decode('utf-8'), '我'.decode('utf-8'), ','.decode('utf-8'),
               '对'.decode('utf-8'), '就'.decode('utf-8'), '中'.decode('utf-8'), '他'.decode('utf-8'), '）'.decode('utf-8'), '（'.decode('utf-8'),
               '-'.decode('utf-8'), '；'.decode('utf-8'), ')'.decode('utf-8'), '.', '('.decode('utf-8'), '？'.decode('utf-8'), '》'.decode('utf-8'),
               '《'.decode('utf-8'), ':', '[', ']'.decode('utf-8'), '！'.decode('utf-8'), '\"', 'NUL', ' '.decode('utf-8'), '', '　 　 '.decode('utf-8'), '☆'.decode('utf-8'),
               '③'.decode('utf-8'), '①'.decode('utf-8'), '②'.decode('utf-8'), '/', '据'.decode('utf-8'), '1', '２'.decode('utf-8'), '3', '4', '5',
               '6', '7', '8', '9', 'Ｇ'.decode('utf-8'), ':', 'the', 'are', 'is', '会'.decode('utf-8'), 'The', '很'.decode('utf-8'), ' 将'.decode('utf-8'),
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

stopWOrdSet = set(stopWordList)

labels = ['A','B','C','D','E','F','G','H','I']

class sogouC(object):
    def __init__(self, root_path, sub_path_list, vocab_mini_count = 5):
        self.root_path = root_path
        self.original_path = os.path.join(root_path, 'original\\')
        self.splited_path = os.path.join(root_path, 'splitted\\')
        self.sub_path_list = sub_path_list
        self.vocab_mini_count = vocab_mini_count

    def split_words(self):
        i = 0
        splited_doc_cache = []
        for sub_path in self.sub_path_list:
            original_path = os.path.join(self.original_path, sub_path)
            splited_path = os.path.join(self.splited_path, sub_path)
            print original_path, 'splitting start...'
            if not os.path.isdir(splited_path):
                os.makedirs(splited_path)
            dirs = os.listdir(original_path)
            j = 1
            for filepath in dirs:
                words = [labels[i]+filepath]
                with open(os.path.join(original_path, filepath), 'r') as f1:
                    lines = f1.readlines()
                    for line in lines:
                        line = line.strip()
                        if line== '': continue
                        words.extend(list(jieba.cut(line.strip())))   #去掉段首的两个空格，并把一篇新闻存成一行。
                splited_doc_cache.append(words)
                j += 1
            i += 1

        with open(os.path.join(root_path, 'splited_text.txt'), 'w') as f2:
            for words in splited_doc_cache:
                line = ''
                for word in words:
                    if word in stopWOrdSet: continue
                    line = line + word + ' '
                f2.write(line + '\n')



    def get_vocab(self):
        vocab_map = dict()
        vocab_set = set()
        for sub_path in self.sub_path_list:
            begin = datetime.now()
            splited_path = os.path.join(self.splited_path, sub_path)
            print splited_path, 'read start...'
            dirs = os.listdir(splited_path)
            for filepath in dirs:
                with open(os.path.join(splited_path, filepath), 'r') as f1:
                    all_the_text = f1.read()
                    words = all_the_text.split(' ')
                    for word in words:
                        word = word.strip()
                        if word not in vocab_map:
                            vocab_map[word] = 0
                        vocab_map[word] += 1
                        vocab_set.add(word)
            end = datetime.now()
            print "time cost is %d second."%((end-begin).seconds)

        vocab_set_sorted = sorted(vocab_map.iteritems(), key=lambda d: d[1], reverse=True)
        vocab_set_valid = []
        with open(os.path.join(root_path, 'dict.txt'), 'w') as f2:
            for word, i in vocab_set_sorted:
                if i > self.vocab_mini_count:
                    vocab_set_valid.append(word)
                    f2.write(word + '\n')

        return vocab_set_valid

    def get_vocab_new(self):
        vocab_map = dict()
        vocab_set = set()
        begin = datetime.now()
        sub_path = 'C0000081'
        splited_path = os.path.join(self.splited_path, sub_path)
        dirs = os.listdir(splited_path)
        for filepath in dirs:
            with open(os.path.join(splited_path, filepath), 'r') as f1:
                all_the_text = f1.read()
                words = all_the_text.split(' ')
                for word in words:
                    word = word.strip()
                    if word not in vocab_map:
                        vocab_map[word] = 0
                    vocab_map[word] += 1
                    vocab_set.add(word)
        end = datetime.now()
        print "time cost is %d ms." % ((end - begin).microseconds/1000)

abs_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
root_path = os.path.join(abs_path, "data\\SogouC.reduced\\")
sub_paths = ['C000008\\', 'C000010\\', 'C000013\\', 'C000014\\', 'C000016\\', 'C000020\\', 'C000022\\', 'C000023\\', 'C000024\\']
print "split words..."
sc = sogouC(root_path, sub_paths, vocab_mini_count=5)
sc.split_words()
#print "get vocab..."
#sc.get_vocab()
#print len(vocab)