#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/19 下午8:24
# @From    : PyCharm
# @File    : main
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import os
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_mnist
from capsNet import CapsNet

def main(_):
    capsNet = CapsNet(is_training=cfg.is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=cfg.logdir,
                             save_model_secs=0)

    path = cfg.results + '/accuracy.csv'
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    elif os.path.exists(path):
        os.remove(path)

    fd_results = open(path, 'w')
    fd_results.write('step, test_acc\n')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:
        num_batch = int(60000 / cfg.batch_size)
        num_test_batch = 10000 // cfg.batch_size
        teX, teY = load_mnist(cfg.dataset, False)
        for epoch in range(cfg.epoch):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                global_step = sess.run(capsNet.global_step)
                sess.run(capsNet.train_op)

                if step % cfg.train_sum_freq == 0:
                    test_acc = 0
                    for i in range(num_test_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        test_acc += sess.run(capsNet.batch_acc,
                                             {capsNet.X: teX[start:end], capsNet.labels: teY[start:end]})
                    test_acc = test_acc / (cfg.batch_size * num_test_batch)
                    fd_results.write(str(global_step + 1) + ',' + str(test_acc) + '\n')
                    fd_results.flush()

            if epoch % cfg.save_freq == 0:
                sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    fd_results.close()
    tf.logging.info('Training done')

if __name__ == "__main__":
    tf.app.run()
