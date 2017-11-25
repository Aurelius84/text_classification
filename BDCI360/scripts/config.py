#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午6:00
# @From    : PyCharm
# @File    : config
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import argparse

parser = argparse.ArgumentParser(description='CNN and RNN model for BDI360')

#########################
# model based parameters
#########################
parser.add_argument('--n-fold', type=int, default=10,
                    help='number of fold to split data for validate')
parser.add_argument('--batch-size', '-b', type=int, default=100,
                    help='batch size')
parser.add_argument('--embed-dim', type=int, default=50,
                    help='embedding dimension')
parser.add_argument('--input-type', type=str, default='char',
                    help='type of input data, word or char')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0,
                    help='clip learning rate')
parser.add_argument('--epochs', type=int, default=2,
                    help='iter number of epochs for training')
parser.add_argument('--global-step', type=int, default=0,
                    help='global step for batch training')
parser.add_argument('--static', type=bool, default=False)
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers to load data for training')
parser.add_argument('--log-interval', type=int, default=50,
                    help='report interval')

#########################
# file path based
#########################
parser.add_argument('--data', type=str, default='../../docs/data',
                    help='dataset dirname')
parser.add_argument('--train-file', type=str, default='train.tsv',
                    help='file name of train dataset')
parser.add_argument('--eval-file', type=str, default='evaluation_public.tsv',
                    help='file name of eval dataset')
parser.add_argument('--embed-path', type=str,
                    help='embedding dimension')
parser.add_argument('--word_vocab', type=int, default=1485695,
                    help='word vocab size')
parser.add_argument('--char_vocab', type=int, default=14804,
                    help='char vocab size')
parser.add_argument('--content-word-seq-len', type=int, default=2619,
                    help='content word sequence length')
parser.add_argument('--content-char-seq-len', type=int, default=3385,
                    help='content char sequence length')

#########################
# gpu parameters
#########################
parser.add_argument('--gpu', type=str, default="0",
                    help='use which gpu to run training, optional 0, 1')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed (default: 10)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

# parser args
params = parser.parse_args()


