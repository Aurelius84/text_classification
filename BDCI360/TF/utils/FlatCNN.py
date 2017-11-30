#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/30 10:12
# @From    : PyCharm
# @File    : FlatCNN
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import tensorflow as tf
from keras.layers import (GRU, Activation, BatchNormalization, Conv1D, Dense,
                          Dropout, Embedding, MaxPool1D, ActivityRegularization, concatenate)
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.objectives import categorical_crossentropy, binary_crossentropy


class FlatCNN(object):

    def __init__(self, params):

        # titles
        self.titles = tf.placeholder(tf.float32, [None, params['title_dim']], name='titles')
        # content
        self.content = tf.placeholder(tf.float32,
                                      [None, params['content_dim']], name='content')
        # labels
        self.labels = tf.placeholder(tf.float32, [None, params['label']['dim']], name='labels')
        # combine_feature
        self.combine_feature = tf.placeholder(tf.float32, [None, params['combine_dim']], name='combine_feature')

        # learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # 1. embedding layers
        # with tf.device('/cpu:0'):
        embedding = Embedding(
            output_dim=params['embed_dim'],
            input_dim=params['vocab_size'],
            input_length=params['content_dim'],
            name="embedding",
            mask_zero=False)(self.content)
        embedding = BatchNormalization(momentum=0.9)(embedding)

        # 2. convolution for content first
        k_s = [2, 3, 4, 5]
        p_s = [2, 2, 3, 3]
        conv_res = []
        for i in range(len(k_s)):
            conv = Conv1D(filters=128, kernel_size=k_s[i])(embedding)
            # batch_norm
            conv = BatchNormalization(momentum=0.9)(conv)
            # activation
            conv = Activation('relu')(conv)

            conv = Conv1D(filters=128, kernel_size=k_s[i])(conv)
            # batch_norm
            conv = BatchNormalization(momentum=0.9)(conv)
            # activation
            conv = Activation('relu')(conv)
            # max_pooling
            conv = MaxPool1D(p_s[i])(conv)

            # 3. Bi-LSTM for outputs of convolution layers
            rnn_cell = GRU(64)(conv)
            rnn_cell = BatchNormalization(momentum=0.9)(rnn_cell)
            conv_res.append(rnn_cell)

        conv_res = concatenate(conv_res, axis=-1)
        # 4. consider combine_feature feature
        combine_layer = concatenate([rnn_cell, self.combine_feature], name='combine_layer')

        # add hidden layer
        combine_layer = Dense(256, name='hidden')(combine_layer)
        combine_layer = BatchNormalization(momentum=0.9)(combine_layer)
        combine_layer = Activation(activation='relu')(combine_layer)

        self.probs = Dense(params['label']['dim'], name='label_probs')(combine_layer)
        self.probs = Activation('sigmoid')(self.probs)


        # 6. calculate loss
        self.preds = tf.argmax(self.probs, axis=1, name="predictions")
        correct_prediction = tf.equal(
            tf.cast(self.preds, tf.int32), tf.cast(tf.argmax(self.labels, axis=1), tf.int32))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="accuracy")

        if 'binary' in params['label']['loss_func']:
            self.loss = tf.reduce_mean(binary_crossentropy(self.labels, self.probs), name='loss')
        else:
            self.loss = tf.reduce_mean(categorical_crossentropy(self.labels, self.probs), name='loss')

        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #     if 'bias' not in v.name]) * 0.01

        # self.loss = tf.add(self.loss, l2_loss)

        # 7. set train_op
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss)


if __name__ == '__main__':
    import yaml
    import numpy as np
    from keras import backend as K

    K.set_learning_phase(1)

    params = yaml.load(open('./params.yaml', 'rb'))
    params['content_dim'] = 100
    params['vocab_size'] = 500
    params['embed_dim'] = 56
    params['combine_dim'] = 3
    x = np.random.randint(0, high=params['vocab_size']-1, size=[10, 100])
    y = np.zeros([10, 2])
    y[: 1] = 1
    cm = np.random.rand(10, 3)

    cnn_rnn = FlatCNN(params)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(100):
            trn_loss, trn_probs, trn_acc, _ = sess.run(
                [
                    cnn_rnn.loss, cnn_rnn.probs, cnn_rnn.accuracy,
                    cnn_rnn.train_op
                ],
                feed_dict={
                    cnn_rnn.content: x,
                    cnn_rnn.labels: y,
                    cnn_rnn.combine_feature: cm,
                    cnn_rnn.learning_rate: 0.001,
                    # K.learning_phase(): 1
                })
            print(trn_loss)