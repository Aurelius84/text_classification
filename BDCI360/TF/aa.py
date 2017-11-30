#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/25 21:36
# @From    : PyCharm
# @File    : aa
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Convolution3D
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.layers import (GRU, Activation, BatchNormalization, Conv1D, Dense, Conv2D,
                          Dropout, Embedding, MaxPool2D, ActivityRegularization, concatenate)

nfilter= 20
kernelsize = 5
space_dim = 200
channel_dim = 3   # like an RGB

inputshape = (10,channel_dim,space_dim,space_dim, 1 )

# initialize the layers with the same filter to check if they actually do the same operation
convW = np.random.rand(kernelsize,kernelsize,1,nfilter)
convB = np.random.rand(nfilter,)

# option 1: TimeDistributed Conv2D
inputs = Input(shape=(channel_dim,space_dim,space_dim, 1, ))
tt = TimeDistributed(Convolution2D(nfilter, kernel_size=(kernelsize, space_dim)))
                           # batch_input_shape=inputshape)
outputs = tt(inputs)
outputs = BatchNormalization(momentum=0.9, axis=-1)(outputs)
outputs = Activation('relu')(outputs)
tm = TimeDistributed(MaxPool2D(kernel_size=(kernelsize, space_dim)))
                           # batch_input_shape=inputshape)
outputs = MaxPool2D(2)
model = Model(inputs=inputs, outputs=outputs)

# model_td.add(TimeDistributed(Convolution2D(nfilter, kernelsize, space_dim, weights=(convW, convB)),
#                           batch_input_shape=inputshape))

# # option 2: Conv3D with 1xKxK kernels
# model_3d = Sequential()
# model_3d.add(Convolution3D(nfilter, 1 , kernelsize, kernelsize, weights=(np.stack([convW]), convB),
#                            batch_input_shape=inputshape))

X = np.random.rand(*inputshape)

r = model.predict(X)
# r2 = model_3d.predict(X)

print(r.shape)