# -*- coding: utf-8 -*-
# @Time    : 2017/12/5 下午11:36
# @Author  : Qi MO
# @Contact  : putao0124@gmail.com
# @Site  : http://blog.csdn.net/coraline_m
# @File    : train.py
# @Software: PyCharm

from keras.utils import plot_model
import numpy as np
from utils.data_helper import load_data_cv
import yaml
import json
from utils.VDCNN import VDCNN
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置gpu限制 参考 http://hpzhao.com/2016/10/29/TensorFlow%E4%B8%AD%E8%AE%BE%E7%BD%AEGPU/
# config = tf.ConfigProto(allow_soft_placement=True)  # 自动选择一个存在并支持的gpu
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# 动态申请gpu，用多少申请多少
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

def generate_arrays(datas, batch_size):

    while 1:
        word_content, labels = [], []
        for i in range(batch_size):
            index = np.random.choice(len(datas),1)
            word_content.append(datas[index[0]].word_content)
            labels.append(datas[index[0]].deal_judge)
        yield (np.array(word_content),np.array(labels))

def train(params):
    train_datas, char_vocab, word_vocab = load_data_cv(
        file_path='../docs/data/train.tsv_cutv0_1_fix_old_bak.csv',
        char_voc_path='../docs/data/char_voc.json',
        word_voc_path='../docs/data/word_voc.json',
        mode='train',
        cv=10)

    params['title_dim'] = len(train_datas[0].deal_title)
    params['content_dim'] = len(train_datas[0].word_content)
    params['combine_dim'] = len(train_datas[0].combine_feature)
    params['vocab_size'] = len(word_vocab)
    # check params on terminal
    print(json.dumps(params, indent=4))

    test_datas, _, _ = load_data_cv(
        file_path='../docs/data/evaluation_public.tsv_cutv0_1_fix.csv',
        char_voc_path='../docs/data/char_voc.json',
        word_voc_path='../docs/data/word_voc.json',
        mode='eval',
        cv=10)

    print('train data size {}'.format(len(train_datas)))
    print('test data size {}'.format(len(test_datas)))

    model = VDCNN(params).model
    hist = model.fit_generator(
        generate_arrays(train_datas, params['batch_size']),
        steps_per_epoch=1000,
        # steps_per_epoch=len(datas_train)/params['batch_size'],
        epochs=params['epoch'],
        verbose=2,  # 2 for one log line per epoch
        validation_data=generate_arrays(test_datas, params['batch_size']),
        validation_steps=1000,
        # validation_steps=len(datas_test)/params['batch_size'],
    )

    model_name = '{}/{}.h5'.format(params['model_dir'], params['model_name'])
    model_plot = '{}/{}.png'.format(params['model_dir'], params['model_name'])
    model_plot_detail = '{}/{}_detail.png'.format(params['model_dir'], params['model_name'])

    plot_model(model, to_file=model_plot)
    plot_model(model, to_file=model_plot_detail, show_shapes=True)
    model.save(model_name)


if __name__ == '__main__':
    # load params
    params = yaml.load(open('./utils/params.yaml', 'r'))

    train(params)