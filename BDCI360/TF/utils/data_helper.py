"""
Data_helper
Competition url: http://www.datafountain.cn/#/competitions/276/intro
"""

# -*- coding: utf-8 -*-
# @Time    : 2017/10/13 下午9:54
# @Author  : Qi MO
# @File    : utils.py
# @Software: PyCharm

import json
import jieba
import os.path
import re
import pickle
import pandas as pd
import time

import numpy as np
from multiprocessing import cpu_count

processnum = 8 if cpu_count() > 8 else 4

# jieba.enable_parallel(processnum)

global split_token
split_token='\u0001'


class ArticleSample(object):
    def __init__(self, _id, title, deal_title, content, deal_content, judge,
                 deal_judge, cv, word_content=None, deal_sentences=None, sent_wd_real_len=None,
                 ):
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
        self.deal_sentences = deal_sentences
        self.sent_wd_real_len = sent_wd_real_len

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

    train = pd.read_table(file_path, sep='\\t', encoding='utf-8', header=None,
                          engine='python')

    char_max_title_length = max(len(x) for x in train[1])
    char_max_content_length = max(len(x) for x in train[2])

    start = time.time()
    title_segment = jieba.cut(split_token.join(train[1]))
    title_segment = ' '.join(title_segment).split(split_token)
    assert len(title_segment) == len(train)
    end = time.time()
    print('segment title complete, cost %.2f s' % (end - start))

    word_max_title_length = max(len(x.split()) for x in title_segment)

    content_segment = jieba.cut(split_token.join(train[2]))
    content_segment = ' '.join(content_segment).split(split_token)
    assert len(content_segment) == len(train)
    print('segment content complete, cost %.2f s' % (time.time() - end))

    word_max_content_length = max(len(x.split()) for x in content_segment)

    char_set = set(''.join(train[1]))
    words_set = set(' '.join(content_segment).split())

    char_voc = {ch: ind + 1 for ind, ch in enumerate(char_set)}
    char_voc['<s>'] = 0

    word_voc = {wd: ind + 1 for ind, wd in enumerate(words_set)}

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
    return [char_voc, char_max_title_length, char_max_content_length, word_voc, word_max_title_length,
            word_max_content_length, title_segment, content_segment]


def load_data_cv(file_path, char_voc_path, word_voc_path, mode, cv=5,
                 min_sen_wd=8, sent_num=100, sent_len=50):
    """
    加载训练数据、测试数据
    :param file_path:  数据地址
    :param char_voc_path:  字的词典地址（建立词典之后直接可以加载）
    :param word_voc_path:  词的词典地址（建立词典之后直接可以加载）
    :param mode:  是训练数据(train)还是测试数据(test)
    :param cv:  几折交叉验证
    :return:
    """
    title_segment, content_segment = None, None

    # 如果已经存在 char 和 word 词典，直接加载
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
    else:  # 否则对原始文件进行词典构建，并返回title和content分词后的结果
        char_voc, char_max_title_length, char_max_content_length, \
        word_voc, word_max_title_length, word_max_content_length, \
        title_segment, content_segment = build_vocab(file_path, char_voc_path, word_voc_path)

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
    pkl_file = file_path[:-4] + '.pkl'
    if os.path.isfile(pkl_file):
        print('load data from temp pkl file {}'.format(pkl_file))
        rev = pickle.load(open(pkl_file, 'rb'))
    else:
        df = pd.read_table(file_path, sep='\\t', encoding='utf-8', header=None,
                           engine='python')
        if mode != 'train':
            df[3] = [''] * len(df)
        print('load data...')

        cnt = 0
        for _id, title, content, judge in zip(df[0], df[1], df[2], df[3]):
            title, content = str(title), str(content)

            # content_word = jieba.lcut(str(content).lower().strip())
            # title = ''.join(title.split())

            sentences = re.split('；。！？!', content)
            sentences = [x for x in sentences if len(x) > min_sen_wd*2]
            # 去掉第一句 和 最后一句
            sentences = sentences[1:-1] if len(sentences) > 5 else sentences
            sent_wd_real_len = [len(x.split()) for x in sentences]
            # wd padding
            sentences = [x.split()[:sent_len] + ['<s>']*max(0, sent_len-len(x.split()))
                         for x in sentences]
            # sent padding
            sentences = sentences[:sent_num] + [['<s>']*sent_len]*max(0, sent_num-len(sentences))
            # index
            sentences = [[word_voc[wd] if wd in word_voc else 0 for wd in sent]
                         for sent in sentences]

            content_word = content.split()
            content = ''.join(content_word)

            pad_title, pad_content = title[:char_max_title_length], content[:char_max_content_length]
            pad_content_word = content_word[:word_max_content_length]
            deal_title = [char_voc[x] if x in char_voc else 0 for x in pad_title]
            deal_title.extend([0] * (char_max_title_length - len(deal_title)))
            deal_content = [char_voc[x] if x in char_voc else 0 for x in pad_content]
            deal_content.extend([0] * (char_max_content_length - len(deal_content)))
            word_content = [word_voc[x] if x in word_voc else 0 for x in pad_content_word]
            word_content.extend([0] * (word_max_content_length - len(pad_content_word)))
            deal_judge = [1, 0] if judge == 'POSITIVE' else [0, 1]
            article = ArticleSample(
                _id=_id,
                title=title,
                deal_title=deal_title,
                content=content,
                deal_content=deal_content,
                word_content=word_content,
                deal_sentences=sentences,
                sent_wd_real_len=sent_wd_real_len,
                judge=judge,
                deal_judge=deal_judge,
                cv=np.random.randint(0, cv))

            rev.append(article)
            # break
            cnt += 1
            if cnt % 2000 == 0:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime())
                print('{} load data: {} ...'.format(timestamp, cnt))
        # 保存中间文件
        pickle.dump(rev, open(pkl_file, 'wb'))
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
    file_path_train = '../../docs/data/add_1000.tsv'
    file_path_test = '../../docs/data/evaluation_public_1000.tsv'
    char_voc_path = '../../docs/data/char_voc.json'
    word_voc_path = '../../docs/data/word_voc.json'
    load_data_cv(file_path_train, char_voc_path, word_voc_path, 'train', max_sen_num=50, min_sen_wd=8)
    # load_data_cv(file_path_test, voc_path, 'eval')
