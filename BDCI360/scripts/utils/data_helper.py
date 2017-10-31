"""
Data_helper
Competition url: http://www.datafountain.cn/#/competitions/276/intro
"""

# -*- coding: utf-8 -*-
# @Time    : 2017/10/13 下午9:54
# @Author  : Qi MO
# @File    : data_helper.py
# @Software: PyCharm

import csv
import json
import jieba
import jieba.posseg as pseg
import os.path
import re
import pickle
import pandas as pd

import numpy as np


class ArticleSample(object):
    def __init__(self, _id, title, deal_title, content, deal_content, word_content, judge,
                 deal_judge, cv):
        """
        文章对象
        :param id:  文章ID(str)
        :param title:  文章标题(str)
        :param deal_title: 处理后的文章标题(list)
        :param content:  文章内容(str)
        :param deal_content:  处理后的文章字表示(list)
        :param word_content:  处理后的文章词表示(list)
        :param judge:  结果(str: NEGATIVE或POSITIVE)
        :param deal_judge: 结果(1表示POSITIVE，0表示NEGATIVE)
        :param cv: 交叉验证标记(int)
        :param content_repeat: 内容重复特征。1重复表示预测为0，0不重复表示预测为1，f1可以达到0.89
                               padding之前做content_repeat
        """
        self.id = _id
        self.title = title
        self.content = content
        self.judge = judge
        self.cv = cv
        self.deal_title = deal_title
        self.deal_content = deal_content
        self.word_content = word_content

        self.deal_judge = deal_judge
        self.content_repeat = 0
        # 真实长度对1000做归一
        self.real_len = len(self.content) / 1000.
        # 是否为短文本，字长小于160的均判断为短文本
        self.is_short = 1 if len(self.content) <= 160 else 0
        self.predict = ''
        self.get_content_repeat()

        # 各种单个特征，如重复、句子长度组合的特征，放在预测的前一层
        self.combine_feature = [self.real_len, self.content_repeat, self.real_len]

    def get_content_repeat(self):
        sen_dict = {}
        sen_list = re.split('，|。', self.content)
        for sen in sen_list:
            sen_dict[sen] = sen_dict.get(sen, 0) + 1
            if (sen_dict[sen] > 1 and (len(sen) >= 10)):
                self.content_repeat = 1
                return



def build_vocab(file_path, char_voc_path, word_voc_path):
    """
    建立词典
    :param file_path: 建立词典文件地址
    :param char_voc_path: 字的词典保存地址
    :param word_voc_path: 词的词典保存地址
    :return:
    """
    print('build vocab...')
    char_voc,word_voc = {'<s>': 0},{'<s>':0}
    char_voc_index,word_voc_index = 0,0
    char_max_title_length,word_max_title_length = 0,0
    char_max_content_length,word_max_content_length = 0,0

    train = pd.read_table(file_path, sep='\\t', encoding='utf-8', header=None,
                          engine='python')
    cnt = 0
    for title, content in zip(train[1],train[2]):
        cnt += 1
        if cnt % 2000 == 0:
            print('build vocab over {} ...'.format(cnt))
        title,content = str(title),str(content)
        char_max_title_length = max(char_max_title_length, len(title))
        char_max_content_length = max(char_max_content_length, len(content))
        for x in title+content:
            if x not in char_voc:
                char_voc_index += 1
                char_voc[x] = char_voc_index
        # words = pseg.cut(str(title).lower().strip())
        # word_title = []
        # for w in words:
        #     if w.flag in ['n', 'nr', 'ns', 'nt', 'nz']:
        #         word_title.append(w.word)
        # words = pseg.cut(str(content).lower().strip())
        # word_content = []
        # for w in words:
        #     if w.flag in ['n', 'nr', 'ns', 'nt', 'nz']:
        #         word_content.append(w.word)
        word_title = jieba.lcut(str(title).lower().strip())
        word_content = jieba.lcut(str(content).lower().strip())
        word_max_title_length = max(word_max_title_length,len(word_title))
        word_max_content_length = max(word_max_content_length, len(word_content))
        word_content.extend(word_title)
        for x in word_content:
            if x not in word_voc:
                word_voc_index +=1
                word_voc[x] = word_voc_index

    print('build vocab done')
    char_voc_dict = {
        'voc': char_voc,
        'max_title_length': char_max_title_length,
        'max_content_length': char_max_content_length
    }
    word_voc_dict = {
        'voc': word_voc,
        'max_title_length': word_max_title_length,
        'max_content_length': word_max_content_length
    }
    with open(char_voc_path, 'w') as f:
        json.dump(char_voc_dict, f)
    with open(word_voc_path, 'w') as f:
        json.dump(word_voc_dict, f)
    return [char_voc, char_max_title_length, char_max_content_length,word_voc, word_max_title_length, word_max_content_length]


def load_data_cv(file_path, char_voc_path,word_voc_path, mode, cv=5):
    """
    加载训练数据、测试数据
    :param file_path:  数据地址
    :param char_voc_path:  字的词典地址（建立词典之后直接可以加载）
    :param word_voc_path:  词的词典地址（建立词典之后直接可以加载）
    :param mode:  是训练数据(train)还是测试数据(test)
    :param cv:  几折交叉验证
    :return:
    """
    if os.path.isfile(char_voc_path):
        with open(char_voc_path, 'r') as f:
            voc_dict = json.load(f)
            char_voc = voc_dict['voc']
            char_max_title_length = voc_dict['max_title_length']
            char_max_content_length = voc_dict['max_content_length']
        with open(word_voc_path, 'r') as f:
            voc_dict = json.load(f)
            word_voc = voc_dict['voc']
            word_max_title_length = voc_dict['max_title_length']
            word_max_content_length = voc_dict['max_content_length']
    else:
        char_voc, char_max_title_length, char_max_content_length,\
        word_voc, word_max_title_length, word_max_content_length\
            = build_vocab(file_path, char_voc_path, word_voc_path)
    char_max_content_length = min(char_max_content_length, 1000)
    word_max_content_length = min(word_max_content_length, 200)
    print('len char voc: ', len(char_voc))
    print('char_max_title_length: ', char_max_title_length)
    print('char_max_content_length: ', char_max_content_length)
    print('len word voc: ', len(word_voc))
    print('word_max_title_length: ', word_max_title_length)
    print('word_max_content_length: ', word_max_content_length)

    # 加载数据对象, 如果存在中间文件，则直接加载
    rev = []
    pkl_file = file_path.split('.')[0] + '.pkl'
    if os.path.isfile(pkl_file):
        rev = pickle.load(pkl_file)
    else:
        df = pd.read_table(file_path, sep='\\t', encoding='utf-8', header=None,
                              engine='python')
        if mode != 'train':
            df[3] = ['']*len(df)
        print('load data...')
        cnt = 0
        for _id, title, content, judge in zip(df[0], df[1], df[2], df[3]):
            title,content = str(title),str(content)
            cnt += 1
            if cnt % 20000 == 0:
                print('load data:...', cnt)

            content_word = jieba.lcut(str(content).lower().strip())
            # content_word = []
            # for w in words:
            #     # if w.flag in ['n','nr','ns','nt','nz']:
            #     content_word.append(w.word)
            # print('content word...')
            # print(content_word)
            pad_title, pad_content = title[:char_max_title_length], content[:char_max_content_length]
            pad_content_word = content_word[:word_max_content_length]
            deal_title = [char_voc[x] if x in char_voc else 0 for x in pad_title]
            deal_title.extend([0] * (char_max_title_length - len(deal_title)))
            deal_content = [char_voc[x] if x in char_voc else 0 for x in pad_content]
            deal_content.extend([0] * (char_max_content_length - len(deal_content)))
            word_content = [word_voc[x] if x in word_voc else 0 for x in pad_content_word]
            word_content.extend([0] * (word_max_content_length-len(pad_content_word)))
            deal_judge = [1, 0] if judge == 'POSITIVE' else [0, 1]
            article = ArticleSample(
                _id=_id,
                title=title,
                deal_title=deal_title,
                content=content,
                deal_content=deal_content,
                word_content=word_content,
                judge=judge,
                deal_judge=deal_judge,
                cv=np.random.randint(0, cv))

            rev.append(article)
            # break
        # 保存中间文件
        pickle.dump(rev, pkl_file)
        print('save rev data in {} ...'.format(pkl_file))

    print('len rev: ', len(rev))
    # print('rev 0...')
    # print(rev[0].title)
    # print(rev[0].deal_title)
    # print(rev[0].content)
    # print(rev[0].deal_content)
    # print(rev[0].word_content)
    # print(rev[0].judge)
    # print(rev[0].deal_judge)
    cnt, all = 0, 0
    for x in rev:
        a = 1 if x.deal_judge == [1, 0] else 0
        b = x.content_repeat
        if a + b == 1:
            cnt += 1
        all += 1.0
    print(cnt, all, cnt / all)
    return rev, char_voc, word_voc


if __name__ == '__main__':
    file_path_train = '../../docs/data/train_1000.tsv'
    file_path_test = '../../docs/data/evaluation_public_1000.tsv'
    char_voc_path = '../../docs/data/char_voc.json'
    word_voc_path = '../../docs/data/word_voc.json'
    load_data_cv(file_path_train, char_voc_path,word_voc_path, 'train')
    # load_data_cv(file_path_test, voc_path, 'eval')