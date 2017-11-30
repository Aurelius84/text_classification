# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 14:40
# @From    : PyCharm
# @File    : rcnn
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import tensorflow as tf
from keras.layers import (GRU, Activation, BatchNormalization, Conv1D, Dense, Conv2D, MaxPool2D,
                          Dropout, Embedding, MaxPool1D, ActivityRegularization, concatenate)
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.objectives import categorical_crossentropy, binary_crossentropy
from keras.layers.wrappers import TimeDistributed


class RCNN(object):
    def __init__(self, params):
        self.params = params

        # content
        self.content = tf.placeholder(tf.int32, shape=[None, params['sent_num'], params['sent_len']], name='content')
        # labels
        self.labels = tf.placeholder(tf.float32, [None, params['label']['dim']], name='labels')
        # combine_feature
        self.combine_feature = tf.placeholder(tf.float32, [None, params['combine_dim']], name='combine_feature')
        # learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
        # real len of content sentences
        self.content_sent_real_len = tf.placeholder(tf.int32, shape=[None], name='sent_real_len')

        # 1. embedding layers
        self.embedding = Embedding(
            output_dim=params['embed_dim'],
            input_dim=params['vocab_size'],
            # input_length=params['sent_num'],
            name="embedding",
            mask_zero=False)
        # cnn and gru on sentence
        self.crnn = self.CRNN_Cell_2(self.content)

        # add hidden layer
        # 4. consider combine_feature feature
        # combine_layer = concatenate([self.crnn, self.combine_feature], name='combine_layer')
        # combine_layer = Dense(256, name='hidden')(combine_layer)
        # combine_layer = BatchNormalization(momentum=0.9)(combine_layer)
        # combine_layer = Activation(activation='relu')(combine_layer)
        combine_layer = self.crnn

        # 5. predict probs for labels
        kwargs = params['label']['kwargs'] if 'kwargs' in params['label'] else {}
        if 'W_regularizer' in kwargs:
            kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
        self.probs = Dense(
            params['label']['dim'],
            # activation='softmax',
            name='label_probs',
            bias_regularizer=l2(0.01),
            **kwargs)(combine_layer)
        # batch_norm
        if 'batch_norm' in params['label']:
            self.probs = BatchNormalization(**params['label']['batch_norm'])(self.probs)
        self.probs = Activation(params['label']['loss_activate'])(self.probs)

        if 'activity_reg' in params['label']:
            self.probs = ActivityRegularization(
                name='label_activity_reg', **params['label']['activity_reg'])(self.probs)

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


    def CRNN_Cell_2(self, input):
        input_dim = input.get_shape().as_list()
        input = tf.reshape(input, [-1, input_dim[1] * input_dim[2]])
        embed_vec = tf.reshape(self.embedding(input), [-1, input_dim[1], input_dim[2], self.params['embed_dim']])
        # print(embed_vec.get_shape())
        # exit()
        # embed_vec = []
        # for sen_i in range(input.get_shape()[1]):
        #     embed_vec.append(self.embedding(input[:, sen_i, :]))
        # embed_vec = tf.stack(embed_vec, 1)
        # print(embed_vec.get_shape())

        # [batch, sen_num, sen_len, embed_dim] -> [batch, sen_num, sen_len, embed_dim, 1]
        input = tf.expand_dims(embed_vec, -1)
        # print(input.get_shape())
        # Conv layer 1

        time_conv1 = TimeDistributed(Conv2D(128, kernel_size=(2, self.params['embed_dim'])))
        # [batch, sen_num, new_len, 1, filters]
        conv1 = time_conv1(input)
        conv1 = BatchNormalization(momentum=0.9, axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)
        # print(conv1.get_shape())
        time_pool1 = TimeDistributed(MaxPool2D(pool_size=(2, 1), strides=1))

        conv1 = time_pool1(tf.transpose(conv1, perm=[0, 1, 2, 4, 3]))
        # print(conv1.get_shape())
        # exit()
        # print(conv1.get_shape())

        # Conv layer 2
        # conv2 = tf.transpose(conv1, perm=[0, 1, 2, 4, 3])
        # print(conv2.get_shape())

        time_conv2 = TimeDistributed(Conv2D(196, kernel_size=(2, 128)))
        conv2 = time_conv2(conv1)
        conv2 = BatchNormalization(momentum=0.9, axis=-1)(conv2)
        # print(conv2.get_shape())

        # conv2 = tf.squeeze(Activation('relu')(conv2), axis=3)
        conv2 = Activation('relu')(conv2)

        # print(conv2.get_shape())

        # conv2_shape = conv2.get_shape().as_list()
        # conv2 = MaxPool2D(pool_size=(3, conv2_shape[-1]), data_format='channels_first')(conv2)
        time_pool2 = TimeDistributed(MaxPool2D(pool_size=(3, 1), strides=(2, 1)))

        conv2 = time_pool2(tf.transpose(conv2, perm=[0, 1, 2, 4, 3]))

        # print(conv2.get_shape())
        # exit()
        # conv2 = tf.squeeze(conv2, axis=-1)
        # first rnn on every conv result
        time_in_sent = TimeDistributed(GRU(256))
        # print(conv2.get_shape())
        sent_gru = time_in_sent(tf.squeeze(conv2, axis=-1))
        # print(sent_gru.get_shape())
        # exit()


        # print(conv2.get_shape())
        rnn_cell = GRU(128, return_sequences=True)(sent_gru)

        batch_range = tf.range(tf.shape(rnn_cell)[0])
        indices = tf.stack([batch_range, self.content_sent_real_len], axis=1)
        # print(indices.get_shape())
        rnn_cell = tf.gather_nd(rnn_cell, indices)
        # print(rnn_cell.get_shape())

        rnn_cell = BatchNormalization(momentum=0.9)(rnn_cell)

        return rnn_cell

    def CRNN_Cell(self, input):
        conv_output = []
        for sen_i in range(input.get_shape()[1]):
            sen_i_vec = self.embedding(input[:, sen_i, :])
            # Conv layer 1
            conv = Conv2D(
                filters=128,
                kernel_size=2,
                padding='VALID',
                # activation='relu',
                strides=1,
                bias_regularizer=l2(0.01))(sen_i_vec)
            conv = Activation('relu')(BatchNormalization(momentum=0.9)(conv))
            conv = MaxPool1D(2)(conv)
            # print(conv.get_shape())

            # Conv layer 2
            conv = Conv1D(
                filters=128,
                kernel_size=3,
                padding='VALID',
                # activation='relu',
                strides=1,
                bias_regularizer=l2(0.01))(conv)
            conv = Activation('relu')(BatchNormalization(momentum=0.9)(conv))
            # print(conv.get_shape())
            conv = MaxPool1D(3)(conv)
            # print(conv.get_shape())
            # print(tf.squeeze(conv).get_shape())
            conv_output.append(tf.squeeze(conv, 1))

        # lstm layer
        conv_output = tf.stack(conv_output, 1)
        rnn_cell = GRU(128, return_sequences=True)(conv_output)

        batch_range = tf.range(tf.shape(rnn_cell)[0])
        indices = tf.stack([batch_range, self.content_sent_real_len], axis=1)
        # print(indices.get_shape())
        rnn_cell = tf.gather_nd(rnn_cell, indices)
        # print(rnn_cell.get_shape())

        rnn_cell = BatchNormalization(momentum=0.9)(rnn_cell)

        return rnn_cell


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
    params['sent_num'] = 30
    params['sent_len'] = 60
    x = np.random.randint(0, high=params['vocab_size']-1, size=[10, 30, 60])
    y = np.zeros([10, 2])
    real_len = np.random.randint(1, high=4, size=[10])
    y[: 1] = 1
    cm = np.random.rand(10, 3)

    cnn_rnn = RCNN(params)

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
                    cnn_rnn.content_sent_real_len:real_len
                    # K.learning_phase(): 1
                })
            print(trn_loss)








