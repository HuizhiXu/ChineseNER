# -*- coding: UTF-8 -*- 
# @Time : 2022/1/6  19:22
# @Author : Huizhi XU


import os
import re

data_path = '/Users/mac/Documents/220_ner'
all_path = '/Users/mac/Documents/220_ner/'
uuids = os.listdir(all_path)
# print(uuids)

# 一共有219篇，八二分，其中train 175篇，test 44篇

#将train的175篇全部集合在一起， Test的44篇全部集合在一起

train_path = '/Users/mac/ChineseNER/test/tk_test.txt'
test_path = '/Users/mac/ChineseNER/test/tk_train.txt'
train_filter_path = '/Users/mac/ChineseNER/test/tk_filter_train.txt'
test_filter_path = '/Users/mac/ChineseNER/test/tk_filter_test.txt'

# for i,uuid in enumerate(uuids):
#     path = os.path.join(data_path, uuid)
#
#     with open(path, 'r') as p:
#         text = p.read()
#
#     if i < 175:
#         with open(train_path, 'a+') as tr:
#             tr.write(text)
#
#     elif 175 <= i < 219:
#         with open(test_path, 'a+') as te:
#             te.write(text)


#对数据进行过滤，只保留中文字符。

pattern = re.compile('[\u4e00-\u9fa5]+')

# with open(test_path, 'r') as file:
#     text = file.read().split('\n')
#
# with open(test_filter_path, 'a+') as tfp:
#     for i, t in enumerate(text):
#         word = t.split('\t')[0]
#         tag = t.split('\t')[1]
#         if pattern.search(word):
#             tfp.write(t)
#             tfp.write('\n')



with open(train_path, 'r') as train_file:
    train_text = train_file.read().split('\n')

with open(train_filter_path, 'a+') as tf_train:
    for i, t in enumerate(train_text):
        word = t.split('\t')[0]
        tag = t.split('\t')[1]
        if pattern.search(word):
            tf_train.write(t)
            tf_train.write('\n')
