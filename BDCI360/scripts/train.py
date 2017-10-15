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
from math import ceil

import numpy as np
import tensorflow as tf
import yaml
from keras import backend as K
from utils.cnn_rnn import CNNRNN
from utils.data_helper import load_data_cv

K.set_learning_phase(1)


def do_eval(sess, model, eval_data, batch_size):
    '''
    eval test data for moedel.
    '''
    K.set_learning_phase(0)
    number_of_data = len(eval_data)
    labels, probs = [], []
    number_of_batch = ceil(number_of_data / batch_size)
    eval_loss, eval_cnt = 0., 0.
    for batch in range(number_of_batch):
        start = batch_size * batch
        end = start + min(batch_size, number_of_data - start)

        curr_titles = [article.deal_title for article in eval_data[start:end]]
        curr_contents = [
            article.deal_content for article in eval_data[start:end]
        ]
        curr_labels = [article.deal_judge for article in eval_data[start:end]]
        curr_content_repeat = [
            article.content_repeat for article in eval_data[start:end]
        ]

        curr_loss, curr_probs = sess.run(
            [model.loss, model.probs],
            feed_dict={
                model.titles: curr_titles,
                model.content: curr_contents,
                model.labels: curr_labels,
                model.content_repeat: curr_content_repeat
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
    print(np.argmax(labels[:3], axis=1) == np.argmax(probs[:3], axis=1))
    acc = np.mean(np.argmax(labels, axis=1) == np.argmax(probs, axis=1))

    K.set_learning_phase(1)

    return eval_loss / eval_cnt, acc


def train(params):
    '''
    train and eval.
    '''
    datas, vocab = load_data_cv(
        file_path='../docs/data/train.tsv',
        voc_path='../docs/data/voc.json',
        mode='train',
        cv=10)

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

    # 设置gpu限制
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    # add model saver, default save lastest 4 model checkpoints
    model_name = params['model_dir'] + params['model_name']

    with tf.Session(config=config) as sess, tf.device('/gpu:1'):
        cnn_rnn = CNNRNN(params)

        saver = tf.train.Saver(max_to_keep=4)
        test_writer = tf.summary.FileWriter(log_test_dir)
        train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        step = -1
        best_acc, best_step = 0., 0
        for epoch in range(params['epoch']):
            # shuffle in each epoch
            train_datas = np.random.permutation(train_datas)

            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                step += 1
                titles = [
                    article.deal_title for article in train_datas[start:end]
                ]
                contents = [
                    article.deal_content for article in train_datas[start:end]
                ]
                labels = [
                    article.deal_judge for article in train_datas[start:end]
                ]
                repeat = [
                    article.content_repeat
                    for article in train_datas[start:end]
                ]

                trn_loss, trn_probs, trn_acc, _ = sess.run(
                    [
                        cnn_rnn.loss, cnn_rnn.probs, cnn_rnn.accuracy,
                        cnn_rnn.train_op
                    ],
                    feed_dict={
                        cnn_rnn.titles: titles,
                        cnn_rnn.content: contents,
                        cnn_rnn.labels: labels,
                        cnn_rnn.content_repeat: repeat
                        # K.learning_phase(): 1
                    })
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime())
                str_loss = '{}:  epoch: {}, step: {} train_loss: {}, train_acc: {}'.format(
                    timestamp, epoch, step, trn_loss, trn_acc)
                print(str_loss)
                # 每 5个 batch 记录一下train loss
                if step % params['log_train_batch'] == 0:
                    train_writer.add_summary(
                        tf.Summary(value=[
                            tf.Summary.Value(
                                tag="loss", simple_value=trn_loss),
                            tf.Summary.Value(
                                tag="accuracy", simple_value=trn_acc)
                        ]),
                        step)
                # 每 70个 batch 评估一下测试集效果
                if step % params['eval_test_batch'] == 0:
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
                    # judge whether test_acc is greater than before
                    if tst_acc >= best_acc:
                        best_step = step
                        best_sess = sess
                        saver.save(
                            best_sess,
                            model_name + '-%s' % tst_acc,
                            global_step=step,
                            write_meta_graph=True)
                        best_acc = tst_acc
                    # 早停止
                    if step - best_step > params['early_stop_eval_n'] * params['eval_test_batch']:
                        test_writer.close()
                        train_writer.close()
                        # predict and save train data
                        predict(
                            best_sess,
                            cnn_rnn,
                            datas,
                            batch_size,
                            save_name='train.csv')
                        # predict and save eval data
                        eval_public, vocab = load_data_cv(
                            file_path='../docs/data/evaluation_public.tsv',
                            voc_path='../docs/data/voc.json',
                            mode='eval',
                            cv=10)
                        predict(
                            best_sess,
                            cnn_rnn,
                            eval_public,
                            batch_size,
                            save_name='eval_public.csv')
                        exit()

        test_writer.close()
        train_writer.close()
        # predict and save train data
        predict(best_sess, cnn_rnn, datas, batch_size, save_name='train.csv')
        # predict and save eval data
        eval_public, vocab = load_data_cv(
            file_path='../docs/data/evaluation_public.tsv',
            voc_path='../docs/data/voc.json',
            mode='eval',
            cv=10)
        predict(
            best_sess,
            cnn_rnn,
            eval_public,
            batch_size,
            save_name='eval_public.csv')


def predict(sess, model, dataset, batch_size, save_name='eval.csv'):
    '''
    predict labels.
    '''
    print('start to predict labels.....')
    K.set_learning_phase(0)
    number_of_data = len(dataset)
    number_of_batch = ceil(number_of_data / batch_size)

    with open('../docs/result/%s' % save_name, 'w') as f:
        for batch in range(number_of_batch):
            print('current process {} -- {}'.format(number_of_batch, batch))
            start = batch_size * batch
            end = start + min(batch_size, number_of_data - start)

            curr_titles = [
                article.deal_title for article in dataset[start:end]
            ]
            curr_contents = [
                article.deal_content for article in dataset[start:end]
            ]

            curr_content_repeat = [
                article.content_repeat for article in dataset[start:end]
            ]

            curr_preds = sess.run(
                model.preds,
                feed_dict={
                    model.titles: curr_titles,
                    model.content: curr_contents,
                    model.content_repeat: curr_content_repeat
                    # K.learning_phase(): 1
                })

            # transform [1] -> 'POSITIVE'
            for i in range(start, end):
                dataset[i].predict = ['POSITIVE',
                                      'NEGATIVE'][curr_preds[i - start]]
                line = '{}\t{}\t{}\t{}\t{}\n'.format(
                    dataset[i].id, dataset[i].title, dataset[i].content,
                    dataset[i].judge, dataset[i].predict)
                f.write(line)

    K.set_learning_phase(1)


def load_predict(model_meta_path,
                 predict_path,
                 save_name='eval_public.csv',
                 mode='eval',
                 batch_size=128):
    '''
    load lastest model and predict datas.
    :return:
    '''
    # 动态申请gpu，用多少申请多少
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint('../docs/model/'))

        # load graph
        graph = tf.get_default_graph()

        # get input placeholder
        tf_title = graph.get_tensor_by_name('titles:0')
        tf_content = graph.get_tensor_by_name('content:0')
        tf_content_repeat = graph.get_tensor_by_name('content_repeat:0')

        tf_preds = graph.get_tensor_by_name('predictions:0')

        model = TFModel(tf_title, tf_content, tf_content_repeat, tf_preds)

        # predict and save eval data
        datas, vocab = load_data_cv(
            file_path=predict_path,
            voc_path='../docs/data/voc.json',
            mode=mode,
            cv=10)
        predict(sess, model, datas, batch_size, save_name=save_name)


class TFModel(object):
    def __init__(self, title, content, content_repeat, preds):
        self.titles = title
        self.content = content
        self.content_repeat = content_repeat
        self.preds = preds


if __name__ == '__main__':
    # load params
    params = yaml.load(open('./utils/params.yaml', 'r'))

    # train(params)

    # 加载模型，进行数据预测
    load_predict(
        model_meta_path='../docs/model/best-0.906323877069-1750.meta',
        predict_path='../docs/data/evaluation_public.tsv',
        save_name='eval_public.csv',
        mode='eval',
        batch_size=128)
