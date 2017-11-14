#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 下午4:40
# @From    : PyCharm
# @File    : generate_sine_wave.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array((range(L))) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1. / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))