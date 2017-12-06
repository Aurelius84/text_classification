# -*- coding: utf-8 -*-
# @Time    : 2017/12/6 上午9:45
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : VDCNN.py
# @Software: PyCharm

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding,Bidirectional,GRU,LSTM
from keras.layers.merge import Concatenate
from keras.layers import Lambda
import keras.backend as K

class VDCNN(object):
    def __init__(self, params):

        content_dim = params['content_dim']
        word_input = Input(shape=(content_dim,))

        # 1. embedding layers
        # with tf.device('/cpu:0'):
        word_embedding = Embedding(
            output_dim=params['embed_dim'],
            input_dim=params['vocab_size'],
            input_length=content_dim,
            name="embedding",
            mask_zero=False)(word_input)

        word_embedding = Dropout(params['drop_out_word_embed'])(word_embedding)

        # 2.多层CNN
        conv_layer_num = params['con_layer']
        for i in range(1, conv_layer_num + 1):
            H_input = word_embedding if i == 1 else conv
            conv = Convolution1D(
                filters=params['filters'],
                kernel_size=params['filter_size'],
                padding=params['padding_mode'],
                strides=params['strides'],
                )(H_input)
            conv = Dropout(params['drop_out_cnn'])(conv)
            if i %2 ==0:
                conv = MaxPooling1D(pool_size=params['pool_size'])(conv)

        # 3. Dense
        conv = Flatten()(conv)
        word_H = Dropout(params['drop_out_cnn_hidden'], name="cnn_hidden")(conv)
        model_output = Dense(params['label_dim'], activation="softmax")(word_H)

        # 4. 构建model
        self.model = Model(inputs=word_input, outputs=model_output)
        self.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])