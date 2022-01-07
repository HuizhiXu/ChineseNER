# -*- coding: utf-8 -*-

import data_format as format
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite

modelname='model/tk_ner.model'
# train_file='example.train'
# test_file='example.test'

train_sents=list()
test_sents=list()

# print(sent2features(train_sents[0])[0])
X_train=[]
y_train=[]
X_test=[]
y_test=[]


def word2features(sent, i):

    word = sent[i][0]
    cuttag = sent[i][1]
    postag = sent[i][2]
    features = [
        'bias',
        'word='+ word,
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'cuttag=' + cuttag,

    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][2]
        cuttag1 = sent[i - 1][1]
        features.extend([
            '-1:word='+word1,
            '-1:postag=' + postag1,
            '-1:cuttag=' + cuttag1,
           # '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][2]
        cuttag1 = sent[i + 1][1]
        features.extend([
            '+1:word=' + word1,
            '+1:postag=' + postag1,
            '+1:cuttag=' + cuttag1,
            #'+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, cuttag, postag, label in sent]


def sent2tokens(sent):
    return [token for token, cuttag, postag, label in sent]



# 加载数据
def load(train_file='test/tk_filter_train.txt',test_file='test/tk_filter_test.txt'):
    global train_sents, test_sents,X_train,y_train,X_test,y_test

    train_sents=format.load_data(train_file)
    test_sents=format.load_data(test_file)

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    return X_train, y_train, X_test, y_test

# 训练
def train(X_train, y_train):

    trainer = pycrfsuite.Trainer()

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # 迭代次数
        # 'max_linesearch':100,
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(modelname)

def tagger(X_test, y_test):

    tagger = pycrfsuite.Tagger()
    # modelname = 'model/ner.model'
    tagger.open(modelname)
    example_sent = test_sents[0]

    # print(' '.join(sent2tokens(example_sent)), end='\n\n')
    # print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    # print("Correct:  ", ' '.join(sent2labels(example_sent)))

    y_pred = [tagger.tag(xseq) for xseq in X_test]

    bio_classification_report(y_test, y_pred)

# 用已训练好的模型进行命名实体识别
# CRF得到的是各个字对应的实体标记分类结果，需要将其格式化为一个个实体
# 目前采用了Boson命名实体语料中的表示方式，用双大括号将命名实体及其类别在原文本里做出标记。
# 例子：
# 输入   "我觉得北京的天气很不错啊。"
# 输出   "我觉得{{location_name:北京}}的天气很不错啊。"
def ner(text):
    tagger = pycrfsuite.Tagger()
    tagger.open(modelname)
    sent=format.split_by_words(text)
    tag_result=tagger.tag(sent2features(sent))
    result_prob = tagger.probability(tag_result)
    text_result=format.format_boson_data_encode(text,tag_result)
    return text_result, result_prob

# 评估系统准确率。
# 评价是按照多分类任务进行的，计算单位是每一个汉字。所以按实体为单位计算的真实F1值可能比该值低一些。
# 需要 scikit-learn 0.15+ 版本，计算多分类任务的评估结果
def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}


    report=classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )
    print(report)


if __name__ == '__main__':


    train_file = 'test/tk_train.txt'
    test_file = 'test/tk_test.txt'
    X_train, y_train, X_test, y_test = load(train_file, test_file)
    print("a")
    # train(X_train, y_train)
    # result, prob = ner('王亚平是中国第一位进入太空的女航天员。2021年9月10号是教师节')
    # print(f"结果为{result}，概率为{prob}")
    #
    # tagger(X_test, y_test)