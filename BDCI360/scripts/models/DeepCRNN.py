#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/26 10:38
# @From    : PyCharm
# @File    : DeepCRNN
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

from .BaseModule import BaseModule
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class DeepCRNN(BaseModule):
    def __init__(self, opt):
        super(DeepCRNN, self).__init__()
        self.model_name = 'DeepCNNBN'

        kernel_sizes = [3, 5, 7]
        pool_sizes = [2, 3, 4]
        conv_hiddens = [100, 128, 196]
        conv_in = [opt.embed_dim] + conv_hiddens[:-1]

        self.opt = opt

        self.encoder = nn.Embedding(opt.char_vocab, opt.embed_dim)


        content_conv = [nn.Sequential(
            nn.Conv1d(in_channels=conv_in[i],
                      out_channels=conv_hiddens[i],
                      kernel_size=kernel_sizes[i]),
            nn.BatchNorm1d(conv_hiddens[i]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_sizes[i])
        ) for i in range(3)]

        self.content_convs = nn.ModuleList(content_conv)

        seq_len = opt.content_char_seq_len
        for i in range(3):
            seq_len = (seq_len - kernel_sizes[i]) // pool_sizes[i]

        self.rnn_hidden_size = 128
        self.rnn_layer = 2
        self.rnn = nn.GRU(conv_hiddens[-1], self.rnn_hidden_size, self.rnn_layer)

        self.fc = nn.Sequential(
            nn.Linear(self.rnn_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        if opt.embed_path:
            self.encoder.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)['vector']))
        else:
            self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, content, hidden):

        h0 = hidden.contiguous()
        # [batch_size, seq_len, embed_dim]
        content = self.encoder(content)

        # permute to [batch_size, embed_size, seq_len]
        content = content.permute(0, 2, 1)
        for cn_conv in self.content_convs:
            content = cn_conv(content)
            print(content.size())


        # [N, C, L] --> [L, N, C]
        content = content.permute(2, 0, 1)
        print(content.size())
        content, hidden = self.rnn(content, h0)
        print(content.size())
        logits = F.sigmoid(self.fc(content[-1]))

        return logits, hidden

    def init_hidden(self, batch_size, use_cuda=False):
        h0 = torch.zeros(self.rnn_layer, batch_size, self.rnn_hidden_size)
        if use_cuda:
            h0 = h0.cuda()
        return Variable(h0)

if __name__ == '__main__':
    from config import params as opt

    opt.content_char_seq_len = 250
    m = DeepCRNN(opt)
    content = Variable(torch.arange(0, 2500).view(10, 250)).long()
    h0 = m.init_hidden(10)
    print(h0.size())
    o, hidden = m(content, h0)
    print(o.size())
    print(hidden.size())
    # print(o.data)