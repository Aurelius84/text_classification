#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/25 下午2:17
# @From    : PyCharm
# @File    : DeepCNNBN
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

from .BaseModule import BaseModule
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class DeepCNNBN(BaseModule):
    def __init__(self, opt):
        super(DeepCNNBN, self).__init__()
        self.model_name = 'DeepCNNBN'

        kernel_sizes = [1, 2, 3, 4]

        self.opt = opt

        self.encoder = nn.Embedding(opt.word_vocab, opt.embed_dim)

        content_word_conv = [nn.Sequential(nn.Conv1d(in_channels=opt.embed_dim,
                                                     out_channels=128,
                                                     kernel_size=kernel_size),
                                                nn.BatchNorm1d(128),
                                                nn.ReLU(inplace=True),

                                                nn.Conv1d(in_channels=128,
                                                          out_channels=256,
                                                          kernel_size=kernel_size),
                                                nn.BatchNorm1d(256),
                                                nn.ReLU(inplace=True),
                                                nn.MaxPool1d(kernel_size=(opt.content_word_seq_len - kernel_size*2 + 2))
                                                )
                                  for kernel_size in kernel_sizes]
        self.content_word_convs = nn.ModuleList(content_word_conv)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes)*256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        if opt.embed_path:
            self.encoder.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)['vector']))


    def forward(self, content):
        # [batch_size, seq_len, embed_dim]
        content = self.encoder(content)

        # permute to [batch_size, embed_size, seq_len]
        content = [cnw_conv(content.permute(0, 2, 1)) for cnw_conv in self.content_word_convs]

        # [N, C, L]
        content = torch.cat((content), dim=1)

        content = content.view(content.size(0), -1)

        logits = F.sigmoid(self.fc(content))

        return logits

if __name__ == '__main__':
    from config import params as opt
    m = DeepCNNBN(opt)
    content = Variable(torch.arange(0, 2500).view(10, 250)).long()

    o = m(content)
    print(o.size())
    print(o.data)


