"""
Contruct CNN ->  RNN models using Keras and tensorflow.
Competition url: http://www.datafountain.cn/#/competitions/276/intro
"""
import tensorflow as tf
from keras.layers import (GRU, Activation, BatchNormalization, Conv1D, Dense,
                          Dropout, Embedding, MaxPool1D, ActivityRegularization, concatenate)
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.objectives import categorical_crossentropy, binary_crossentropy


class CNNRNN(object):

    def __init__(self, params):

        # with tf.device('/gpu:1'):
        # titles
        self.titles = tf.placeholder(tf.float32, [None, params['title_dim']], name='titles')
        # content
        self.content = tf.placeholder(tf.float32,
                                      [None, params['content_dim']], name='content')
        # labels
        self.labels = tf.placeholder(tf.float32, [None, params['label']['dim']], name='labels')
        # combine_feature
        self.combine_feature = tf.placeholder(tf.float32, [None, params['combine_dim']], name='combine_feature')

        # 1. embedding layers
        # with tf.device('/cpu:0'):
        embedding = Embedding(
            output_dim=params['embed_dim'],
            input_dim=params['vocab_size'],
            input_length=params['content_dim'],
            name="embedding",
            mask_zero=False)(self.content)
        embedding = BatchNormalization(momentum=0.9)(embedding)

        # with tf.device('/cpu:0'):
        # 2. convolution for content first
        conv_layer_num = len(params['Conv1D'])
        for i in range(1, conv_layer_num + 1):
            H_input = embedding if i == 1 else H
            conv = Conv1D(
                filters=params['Conv1D']['layer%s' % i]['filters'],
                kernel_size=params['Conv1D']['layer%s' % i]['filter_size'],
                padding=params['Conv1D']['layer%s' % i]['padding_mode'],
                # activation='relu',
                strides=1,
                bias_regularizer=l2(0.01))(H_input)
            # batch_norm
            if 'batch_norm' in params['Conv1D']['layer%s' % i]:
                conv = Activation('relu')(BatchNormalization(momentum=params['Conv1D']['layer%s' % i]['batch_norm'])(conv))
            H = MaxPool1D(
                pool_size=params['Conv1D']['layer%s' %
                                           i]['pooling_size'])(conv)
            # dropout
            if 'dropout' in params['Conv1D']['layer%s' % i]:
                H = Dropout(
                    params['Conv1D']['layer%s' % i]['dropout'])(H)

        # 3. Bi-LSTM for outputs of convolution layers
        rnn_cell = Bidirectional(
            GRU(params['RNN']['cell'],
                dropout=params['RNN']['dropout'],
                recurrent_dropout=params['RNN']['recurrent_dropout']), merge_mode=params['RNN']['merge_mode'])(H)

        # 4. consider combine_feature feature
        # combine_layer = concatenate([rnn_cell, self.combine_feature], name='combine_layer')
        combine_layer = rnn_cell

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

        # 7. set train_op
        self.train_op = tf.train.RMSPropOptimizer(params['learn_rate']).minimize(
            self.loss)
