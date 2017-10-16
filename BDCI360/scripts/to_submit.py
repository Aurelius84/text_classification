# -*- coding: utf-8 -*-
# @Time    : 2017/10/16 上午10:41
# @Author  : Qi MO
# @File    : to_submit.py
# @Software: PyCharm

import pandas as pd
import numpy as np

file_dir = '../docs/result/'
suffix = '_10_16'

train  = pd.read_table('{}train{}.csv'.format(file_dir,suffix), sep='\\t',encoding='utf-8',header=None, engine='python')
test  = pd.read_table('{}eval_public{}.csv'.format(file_dir,suffix), sep='\\t',encoding='utf-8',header=None, engine='python')

df=train
print('train num: ',len(df))
Precision = np.sum((df[3]=='POSITIVE')&(df[4]=='POSITIVE'))/float(np.sum(df[4]=='POSITIVE'))
Recall = np.sum((df[3]=='POSITIVE')&(df[4]=='POSITIVE'))/float(np.sum(df[3]=='POSITIVE'))

print('precison: ',Precision)
print('recall: ',Recall)
print('F1: ',2.0*Precision*Recall/(Precision+Recall))

df2 = pd.DataFrame()
df2['文章ID'] = test[0]
df2['预测标签'] = test[4]
df2.to_csv('{}res{}.csv'.format(file_dir,suffix),index=False)
