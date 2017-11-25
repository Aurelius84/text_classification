#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午4:24
# @From    : PyCharm
# @File    : cnn_rnn
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNNRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_type):
        super(CNNRNN, self).__init__()
        self.rnn_type = rnn_type

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.layer1 = nn.Sequential(
            nn.Conv1d(embed_dim, 48, kernel_size=5, stride=1),
            nn.MaxPool1d(5, stride=2),
            nn.BatchNorm2d(48),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=7, stride=2),
            nn.MaxPool1d(7, stride=4),
            nn.BatchNorm1d(96),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=9, stride=2),
            nn.MaxPool1d(9, stride=4),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )


        # if rnn_type in ['LSTM', 'GRU']:
        #     self.rnn = getattr(nn, rnn_type)(128, 128, 2, batch_first=True)
        # else:
        #     try:
        #         nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        #     except KeyError:
        #         raise ValueError("""An invalid option for `--model` was supplied,
        #                                  options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        #     self.rnn = nn.RNN(128, 128, 2, nonlinearity=nonlinearity, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128, 2),
            nn.Dropout(0.5),
            # nn.Sigmoid(),
            # nn.LogSoftmax()
        )
        self.init_weights()


    def forward(self, content, hidden, title=None):
        batch_size = content.size(0)

        h0 = hidden.contiguous()
        # embedding layer
        content = self.embed(content)
        # [batch, seq, 1, dim] -> [batch, dim, seq]
        content = content.squeeze().transpose(2, 1)
        # conv layer
        content = self.layer1(content)
        content = self.layer2(content)
        content = self.layer3(content)
        # rnn layer
        # content = content.transpose(1, 2)
        # content, hidden = self.rnn(content, h0)
        output = self.fc(content[:, :, -1])

        return F.softmax(output), hidden

    def init_weights(self, initrange=0.1):

        self.embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, use_cuda=False):
        h0 = torch.zeros(2, batch_size, 128)
        c0 = torch.zeros(2, batch_size, 128)
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        if self.rnn_type == 'LSTM':
            return (Variable(h0), Variable(c0))
        else:
            return Variable(h0)
