#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午6:29
# @From    : PyCharm
# @File    : data_helper
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd
import re
import transforms
import os
import json

class BDCIDataset(Dataset):

    def __init__(self, csv_file,
                 char_voc,
                 word_voc,
                 char_title_len=40,
                 char_content_len=1000,
                 word_title_len=15,
                 word_content_len=500,
                 ):
        self.df = pd.read_table(csv_file, sep='\\t', encoding='utf-8', header=None, engine='python')

        self.ids = self.df[0]
        self.titles = self.df[1]
        self.contents = self.df[2]

        self.labels = self.df[3] if self.df.shape[1] >= 4 else None

        # sequence length of title in char level
        self.char_title_len = char_title_len
        # sequence length of title in word level
        self.word_title_len = word_title_len

        # sequence length of content in char level
        self.char_content_len = char_content_len
        # sequence length of content in word level
        self.word_content_len = word_content_len

        self.char_content_transform = transforms.Compose([
            transforms.ToIndex(char_voc),
            transforms.Pad(self.char_content_len),
            torch.LongTensor
        ])

        self.word_content_transform = transforms.Compose([
            transforms.ToIndex(word_voc),
            transforms.Pad(self.word_content_len),
            torch.LongTensor
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = ArticleSample(id=str(self.ids[idx]),
                               title=str(self.titles[idx]),
                               content=str(self.contents[idx]),
                               label=self.labels[idx] if self.labels is not None else None)
        # sample.content_word = self.contents[idx].split()
        # sample.content_word = list(jieba.cut(str(self.contents[idx])))
        # sample.content_word_vec = self.word_content_transform(sample.content_word)
        sample.content_char_vec = self.char_content_transform(sample.content)
        sample.label_vec = torch.FloatTensor(sample.label_vec)

        return sample.__dict__


class ArticleSample(object):
    def __init__(self,
                 id,
                 title,
                 content,
                 label='',
                 title_word=[],
                 content_word=[],
                 title_char_vec=[],
                 title_word_vec=[],
                 content_char_vec=[],
                 content_word_vec=[],
                 cv=0):

        self.id = id
        self.title = title
        self.content = content
        self.label = label
        self.cv = cv

        self.title_word = title_word
        self.content_word = content_word

        self.title_char_vec = title_char_vec
        self.title_word_vec = title_word_vec

        self.content_char_vec = content_char_vec
        self.content_word_vec = content_word_vec
        if label == '' or label is None:
            self.label_vec = None
        else:
            self.label_vec = [0, 1] if label == 'POSITIVE' else [1, 0]

        '''
        self.content_repeat = 0
        # 真实长度对1000做归一
        self.real_len = len(self.content) / 1000.
        # 是否为短文本，字长小于160的均判断为短文本
        self.is_short = 1 if len(self.content) <= 160 else 0
        self.predict = ''
        self.get_content_repeat()

        # 各种单个特征，如重复、句子长度组合的特征，放在预测的前一层
        self.combine_feature = [self.real_len, self.content_repeat, self.real_len]
        '''

    def get_content_repeat(self):
        sen_dict = {}
        sen_list = re.split('，|。', self.content)
        for sen in sen_list:
            sen_dict[sen] = sen_dict.get(sen, 0) + 1
            if sen_dict[sen] > 1 and (len(sen) >= 10):
                self.content_repeat = 1
                return

def load_voc(dir_path):
    '''
    load char vocab and word vocab
    Args:
        dir_path: str, dir name of vocab

    Returns: two dict of char and word vocab

    '''

    with open(os.path.join(dir_path, 'char_voc.json'), 'r') as f:
        char_dict = json.load(f)
        char_voc = char_dict['voc']
        char_max_title_length = char_dict['max_title_length']
        char_max_content_length = char_dict['max_content_length']
    with open(os.path.join(dir_path, 'word_voc.json'), 'r') as f:
        word_dict = json.load(f)

    return char_dict, word_dict


if __name__ == '__main__':
    char_voc_path = '../../docs/data/char_voc.json'
    word_voc_path = '../../docs/data/word_voc.json'

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

    train = BDCIDataset('../../docs/data/add_1000.tsv',
                        char_voc,
                        word_voc
                        )
    dataloader = DataLoader(train, batch_size=4, shuffle=True)

    for i_batch, sampled_batched in enumerate(dataloader):
        print(i_batch, sampled_batched['label_vec'].size())
        print(sampled_batched['label_vec'])
        exit()