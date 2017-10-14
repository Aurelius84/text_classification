# -*- coding:utf-8 -*-
"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: Atom
@file: adios_train.py
@time: 17/10/13 20:39
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
import yaml
from keras import backend as K
from utils.cnn_rnn import CNNRNN
from utils.data_helper import load_data_cv


def do_eval(sess, model, eval_data, batch_size):
    '''
    eval test data for moedel.
    '''
    K.set_learning_phase(0)
    number_of_data = len(eval_data)
    labels, probs = [], []
    eval_loss, eval_cnt = 0., 0.
    for start, end in zip(
            range(0, number_of_data, batch_size),
            range(batch_size, number_of_data, batch_size)):
        curr_titles = [article.deal_title for article in eval_data[start:end]]
        curr_contents = [article.deal_content for article in eval_data[start:end]]
        curr_labels = [
            article.deal_judge for article in eval_data[start:end]
        ]

        curr_loss, curr_probs = sess.run(
            [model.loss, model.probs],
            feed_dict={
                model.titles: curr_titles,
                model.content: curr_contents,
                model.labels: curr_labels
                # K.learning_phase(): 1
            })
        eval_loss += curr_loss
        eval_cnt += 1

        labels.extend(curr_labels)
        probs.extend(curr_probs)

    # evaluation metrics
    labels = np.array(labels)
    probs = np.array(probs)
    for check_data in zip(labels[:5], probs[:5]):
        print('label:', check_data[0])
        print('probs:', check_data[1])
        print('\n')

    acc = np.mean(np.argmax(labels) == np.argmax(probs))

    K.set_learning_phase(1)

    return eval_loss / eval_cnt, acc


def train(params):
    '''
    train and eval.
    '''
    datas, vocab = load_data_cv(file_path='../docs/data/train_1000.tsv', voc_path='../docs/data/voc.json', mode='train', cv=5)

    params['title_dim'] = len(datas[0].deal_title)
    params['content_dim'] = len(datas[0].deal_content)
    params['vocab_size'] = len(vocab)
    # check params on terminal
    print(json.dumps(params, indent=4))

    test_datas = list(filter(lambda data: data.cv == 1, datas))
    train_datas = list(filter(lambda data: data.cv != 1, datas))

    print('train dataset: {}'.format(len(train_datas)))
    print('test dataset: {}'.format(len(test_datas)))

    number_of_training_data = len(train_datas)
    # build model
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # log for tensorboard visualization
    log_test_dir = '../docs/test/%s' % timestamp
    log_train_dir = '../docs/train/%s' % timestamp
    os.mkdir(log_test_dir)
    os.mkdir(log_train_dir)

    batch_size = params['batch_size']
    with tf.Session() as sess:
        cnn_rnn = CNNRNN(params)
        test_writer = tf.summary.FileWriter(log_test_dir, sess.graph)
        train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        step = -1
        for epoch in range(params['batch_size']):
            # shuffle in each epoch
            train_datas = np.random.permutation(train_datas)

            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                step += 1
                titles = [article.deal_title for article in train_datas[start:end]]
                contents = [article.deal_content for article in train_datas[start:end]]
                labels = [article.deal_judge for article in train_datas[start:end]]

                trn_loss, trn_probs, trn_acc, _ = sess.run(
                    [
                        cnn_rnn.loss, cnn_rnn.probs, cnn_rnn.accuracy,
                        cnn_rnn.train_op
                    ],
                    feed_dict={
                        cnn_rnn.titles: titles,
                        cnn_rnn.content: contents,
                        cnn_rnn.labels: labels
                        # K.learning_phase(): 1
                    })
                # 每 5个 batch 评估一下测试集效果
                if step % 5 == 0:
                    train_writer.add_summary(
                        tf.Summary(value=[
                            tf.Summary.Value(
                                tag="loss", simple_value=trn_loss),
                            tf.Summary.Value(
                                tag="accuracy", simple_value=trn_acc)
                        ]),
                        step)
                    # loss and acc on eval dataset
                    tst_loss, tst_acc = do_eval(sess, cnn_rnn, test_datas,
                                                batch_size)

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime())
                    str_loss = '{}:  epoch: {} eval_loss: {}, eval_acc: {}'.format(
                        timestamp, epoch, tst_loss, tst_acc)
                    print(str_loss)
                    # log to tensorboard
                    test_writer.add_summary(
                        tf.Summary(value=[
                            tf.Summary.Value(
                                tag="loss", simple_value=tst_loss),
                            tf.Summary.Value(
                                tag="accuracy", simple_value=tst_acc)
                        ]),
                        step)
        test_writer.close()
        train_writer.close()


if __name__ == '__main__':
    # load params
    params = yaml.load(open('./utils/params.yaml', 'r'))

    train(params)
