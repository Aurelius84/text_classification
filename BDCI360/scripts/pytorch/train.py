#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午6:00
# @From    : PyCharm
# @File    : train
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import params
from cnn_rnn import CNNRNN
from data_helper import BDCIDataset, load_voc

from tensorboardX import SummaryWriter


FloatTensor = torch.FloatTensor

def train(params, dataloader):

    train_writer = SummaryWriter()

    model = CNNRNN(vocab_size=len(word_dict['voc']) + 1,
                   embed_dim=params.embed_dim,
                   rnn_type='GRU')
    if params.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=params.lr)
    for epoch in range(params.epochs):
        for batch_idx, samples in enumerate(dataloader):

            vx, vy = samples['content_word_vec'], samples['label_vec']
            batch_size = vx.size()[0]
            # init hidden
            init_hidden = model.init_hidden(batch_size, use_cuda=params.cuda)

            if params.cuda:
                vx, vy = samples['content_word_vec'].cuda(), samples['label_vec'].cuda()
            vx, vy = Variable(vx), Variable(vy)

            outputs, hidden = model(vx, init_hidden)

            if params.cuda:
                outputs = outputs.cuda()

            # print(vy.size(), outputs.size())
            loss = F.binary_cross_entropy(vy, outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            # train accuracy
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(outputs.data)
            # print(pred)
            # print(vy.data)

            correct = pred.eq(vy.data.max(1, keepdim=True)[1]).cuda().sum()
            # print(correct)
            # print(loss.data[0])

            if batch_idx % params.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} acc: {:.3f}%'.format(
                    epoch, batch_idx * len(samples), len(dataloader.dataset),
                           100. * batch_idx / len(dataloader), loss.data[0], 100. * correct / batch_size))
                train_writer.add_scalar('loss/train', loss.data[0], global_step=params.global_step)
                train_writer.add_scalar('accuracy/train', correct, global_step=params.global_step)
                params.global_step += 1

    train_writer.close()

def eval():
    pass

def predict():
    pass


if __name__ == '__main__':
    char_dict, word_dict = load_voc('../../docs/data/')
    trainset = BDCIDataset('../../docs/data/add_1000.tsv',
                           char_dict['voc'],
                           word_dict['voc'],
                           char_title_len=char_dict['max_title_length'],
                           char_content_len=char_dict['max_content_length'],
                           word_title_len=word_dict['max_title_length'],
                           word_content_len=word_dict['max_content_length'],
                        )
    trn_dataloader = DataLoader(trainset,
                                batch_size=params.batch_size,
                                shuffle=True, num_workers=8)
    train(params, trn_dataloader)

