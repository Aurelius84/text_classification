#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/9 下午5:00
# @From    : PyCharm
# @File    : data.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        '''
        添加新词
        :param word: 新词
        :return: 新词索引
        '''
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    '''
    文本语料类
    '''
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        '''
        Tokenize a text file
        :param path:
        :return:
        '''
        assert os.path.exists(path)

        # add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for wd in words:
                    self.dictionary.add_word(wd)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids