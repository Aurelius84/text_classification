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
import os.path
import re
import pandas as pd

import numpy as np


class ArticleSample(object):
    def __init__(self, _id, title, deal_title, content, deal_content, judge,
                 deal_judge, cv):
        """
        文章对象
        :param id:  文章ID(str)
        :param title:  文章标题(str)
        :param deal_title: 处理后的文章标题(list)
        :param content:  文章内容(str)
        :param deal_content:  处理后的文章内容(list)
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



def build_vocab(file_path, voc_path):
    """
    建立词典
    :param file_path: 建立词典文件地址
    :param voc_path: 词典保存地址
    :return:
    """
    print('build vocab...')
    voc = {'<s>': 0}
    voc_index = 0
    max_title_length = 0
    max_content_length = 0

    train = pd.read_table(file_path, sep='\\t', encoding='utf-8', header=None,
                          engine='python')

    for title, content in zip(train[1],train[2]):
        title,content = str(title),str(content)
        max_title_length = max(max_title_length, len(title))
        max_content_length = max(max_content_length, len(content))
        # if len(content) == 93749:
        #     print(title)
        #     print(content)
        for x in title:
            if x not in voc:
                voc_index += 1
                voc[x] = voc_index
        for x in content:
            if x not in voc:
                voc_index += 1
                voc[x] = voc_index

    print('build vocab done')
    voc_dict = {
        'voc': voc,
        'max_title_length': max_title_length,
        'max_content_length': max_content_length
    }
    with open(voc_path, 'w') as f:
        json.dump(voc_dict, f)
    return [voc, max_title_length, max_content_length]


def load_data_cv(file_path, voc_path, mode, cv=5):
    """
    加载训练数据、测试数据
    :param file_path:  数据地址
    :param voc_path:  词典地址（建立词典之后直接可以加载）
    :param mode:  是训练数据(train)还是测试数据(test)
    :param cv:  几折交叉验证
    :return:
    """
    rev = []
    if os.path.isfile(voc_path):
        with open(voc_path, 'r') as f:
            voc_dict = json.load(f)
            voc = voc_dict['voc']
            max_title_length = voc_dict['max_title_length']
            max_content_length = voc_dict['max_content_length']
    else:
        voc, max_title_length, max_content_length = build_vocab(
            file_path, voc_path)
    max_content_length = min(max_content_length, 1000)
    print('len voc: ', len(voc))
    print('max_title_length: ', max_title_length)
    print('max_content_length: ', max_content_length)

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

        pad_title, pad_content = title[:
                                       max_title_length], content[:
                                                                  max_content_length]
        deal_title = [voc[x] if x in voc else 0 for x in pad_title]
        deal_title.extend([0] * (max_title_length - len(deal_title)))
        deal_content = [voc[x] if x in voc else 0 for x in pad_content]
        deal_content.extend([0] * (max_content_length - len(deal_content)))
        deal_judge = [1, 0] if judge == 'POSITIVE' else [0, 1]
        article = ArticleSample(
            _id=_id,
            title=title,
            deal_title=deal_title,
            content=content,
            deal_content=deal_content,
            judge=judge,
            deal_judge=deal_judge,
            cv=np.random.randint(0, cv))

        rev.append(article)

    print('len rev: ', len(rev))
    # print('rev 0...')
    # print(rev[0].title)
    # print(rev[0].deal_title)
    # print(rev[0].content)
    # print(rev[0].deal_content)
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
    return rev, voc


if __name__ == '__main__':
    file_path_train = '../../docs/data/train.tsv'
    file_path_test = '../../docs/data/evaluation_public.tsv'
    voc_path = '../../docs/data/voc.json'
    load_data_cv(file_path_train, voc_path, 'train')
    load_data_cv(file_path_test, voc_path, 'eval')