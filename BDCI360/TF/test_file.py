#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午2:34
# @From    : PyCharm
# @File    : test_file.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import jieba
import pandas as pd

jieba.enable_parallel()

train = pd.read_table('../docs/data/train.tsv', sep='\\t', encoding='utf-8', header=None,
                          engine='python')

content = ' '.join(train[1])
words = list(jieba.cut(content))
char = list(set(content))


print(words[:2])
print(len(words))

print(char[:10])
print(len(char))
