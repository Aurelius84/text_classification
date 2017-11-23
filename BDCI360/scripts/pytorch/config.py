#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午6:00
# @From    : PyCharm
# @File    : config
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import argparse

parser = argparse.ArgumentParser(description='CNN and RNN model for BDI360')
parser.add_argument('--data', type=str, default='../../docs/data', help='dataset dirname')
parser.add_argument('--train-file', type=str, default='train.tsv',
                    help='file name of train dataset')
parser.add_argument('--eval-file', type=str, default='evaluation_public.tsv',
                    help='file name of eval dataset')
parser.add_argument('--n-fold', type=int, default=10,
                    help='number of fold to split data for validate')
parser.add_argument('--batch-size', '-b', type=int, default=16, help='batch size')
parser.add_argument('--embed-dim', type=int, default=100, help='embedding dimension')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=6, help='iter number of epochs for training')
parser.add_argument('--global-step', type=int, default=0)
parser.add_argument('--gpu', type=int, default=1, help='use which gpu to run training, optional 0 or 1')
parser.add_argument('--seed', type=int, default=10, help='random seed (default: 10)')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the model')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, help='report interval')

params = parser.parse_args()