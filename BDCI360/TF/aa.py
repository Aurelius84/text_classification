#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/25 21:36
# @From    : PyCharm
# @File    : aa
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
from tqdm import tqdm

import time
for j in range(3):
    for i in tqdm(range(100)):
        time.sleep(0.01)
