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
            nn.Conv1d(embed_dim, 128, kernel_size=3, stride=1),
            nn.MaxPool1d(3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=1),
            nn.MaxPool1d(3, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.MaxPool1d(3, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )


        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(512, 1024, 2, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(512, 1024, 2, nonlinearity=nonlinearity, batch_first=True)

        self.fc = nn.Linear(1024 * 226, 2)

    def forward(self, content, hidden, title=None):

        h0 = hidden.contiguous()
        # embedding layer
        content = self.embed(content)
        # [batch, seq, 1, dim] -> [batch, dim, seq]
        content = content.view(content.size()[0], content.size()[-1], -1)
        # conv layer
        content = self.layer1(content)
        content = self.layer2(content)
        content = self.layer3(content).squeeze()
        # rnn layer
        content = content.transpose(1, 2)
        content, hidden = self.rnn(content, h0)

        content = content.contiguous().view(content.size()[0], -1)
        output = self.fc(content)

        return F.softmax(output), hidden

    def init_weights(self, initrange=0.1):

        self.embed.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size, use_cuda=False):
        h0 = Variable(torch.zeros(2, batch_size, 1024))
        c0 = Variable(torch.zeros(2, batch_size, 1024))
        if use_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        if self.rnn_type == 'LSTM':
            return (h0, c0)
        else:
            return h0
