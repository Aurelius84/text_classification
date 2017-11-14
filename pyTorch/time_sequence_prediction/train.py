#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 下午4:40
# @From    : PyCharm
# @File    : train.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=True)
        c_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=True)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    # load data
    data = torch.load('traindata.pt')
    input = Variable(torch.from_numpy(data[3:, :-1]), requires_grad=False)
    target = Variable(torch.from_numpy(data[3:, 1:]), requires_grad=False)

    test_input = Variable(torch.from_numpy(data[:3, :-1]), requires_grad=False)
    test_target = Variable(torch.from_numpy(data[:3, 1:]), requires_grad=False)

    # build_model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss: ', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)

        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss: ', loss.data.numpy()[0])
        y = pred.data.numpy()

        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)


        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf' % i)
        plt.show()
        plt.close()
