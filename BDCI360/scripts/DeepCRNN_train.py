#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/26 13:21
# @From    : PyCharm
# @File    : DeepCRNN_train
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

import models
from config import params
from utils import BDCIDataset, load_voc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu

global input_type
# content_word_vec or content_char_vec
input_type = 'content_' + params.input_type + '_vec'


def train(dataloader):
    '''
    train model with data from dataloader
    Args:
        dataloader: implement from torch.utils.data.DataLoader, batch output from iterators.

    Returns: saved model and tensorboard logs

    '''
    train_writer = SummaryWriter(comment='_train')
    global test_writer
    test_writer = SummaryWriter(comment='_test')

    model = models.DeepCRNN(params)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = torch.nn.BCELoss()
    if params.cuda:
        model.cuda()
        criterion.cuda()

    min_loss = 1000
    model.train()

    for epoch in range(params.epochs):
        # switch to train model
        batch_idx = 0
        for samples in tqdm(dataloader):
            batch_idx += 1
            # print(samples['title'][0], samples['label'][0])
            vx, vy = samples[input_type], samples['label_vec']
            batch_size = vx.size()[0]
            h0 = model.init_hidden(batch_size, use_cuda=params.cuda)

            if params.cuda:
                vx, vy = samples[input_type].cuda(), samples['label_vec'].cuda()
            vx, vy = Variable(vx), Variable(vy)
            outputs, hidden = model(vx, h0)

            # if params.cuda:
            #     outputs = outputs.cuda()

            optimizer.zero_grad()
            loss = criterion(outputs, vy)
            loss.backward()
            if params.clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
            optimizer.step()

            # train accuracy
            pred = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(vy.data.max(1, keepdim=True)[1]).cuda().sum()
            acc = correct / batch_size

            if batch_idx % params.log_interval == 0:
                train_writer.add_scalar('loss', loss.data[0], global_step=params.global_step)
                train_writer.add_scalar('accuracy', acc, global_step=params.global_step)
                params.global_step += 1
                # 评估模型
                test_loss, test_acc = val(model, eval_dataloader, params.global_step)
                if test_loss < min_loss:
                    # save model
                    model.save(name=params.save_name)
                    min_loss = test_loss
                else:
                    optimizer = model.get_optimizer(lr1=params.lr/2, lr2=params.lr*0.8)

    train_writer.close()


def val(model, dataloader, step):
    '''
    validate model using valid dataset.
    Args:
        model: training model
        dataloader: implement from torch.utils.data.DataLoader, batch output from iterators.
        step: global step for log information on tensorboard.

    Returns:

    '''
    model.eval()
    test_loss, correct = 0., 0
    for samples in tqdm(dataloader):
        data, target = samples[input_type], samples['label_vec']
        batch_size = data.size()[0]
        h0 = model.init_hidden(batch_size, use_cuda=params.cuda)
        if params.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)

        output, hidden = model(data, h0)
        if params.cuda:
            output = output.cuda()
        test_loss += F.binary_cross_entropy(output, target, size_average=False).data[0]  # sum up batch loss

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.max(1, keepdim=True)[1]).cuda().sum()

    test_loss /= len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)

    test_writer.add_scalar('loss', test_loss, global_step=step)
    test_writer.add_scalar('accuracy', acc, global_step=step)
    return test_loss, acc

def predict():
    pass


if __name__ == "__main__":
    char_dict, word_dict = load_voc('../docs/data/')
    # 训练集文本
    trainset = BDCIDataset(os.path.join(params.data, params.train_file),
                           char_dict['voc'],
                           word_dict['voc'],
                           char_title_len=char_dict['max_title_length'],
                           char_content_len=char_dict['max_content_length'],
                           word_title_len=word_dict['max_title_length'],
                           word_content_len=word_dict['max_content_length'],
                           )
    trn_dataloader = DataLoader(trainset,
                                pin_memory=True,
                                batch_size=params.batch_size,
                                shuffle=True, num_workers=params.num_workers)
    # 测试集文本
    eval_set = BDCIDataset(os.path.join(params.data, params.eval_file),
                           char_dict['voc'],
                           word_dict['voc'],
                           char_title_len=char_dict['max_title_length'],
                           char_content_len=char_dict['max_content_length'],
                           word_title_len=word_dict['max_title_length'],
                           word_content_len=word_dict['max_content_length'],
                           )
    eval_dataloader = DataLoader(eval_set,
                                 pin_memory=True,
                                 batch_size=params.batch_size,
                                 shuffle=False, num_workers=params.num_workers)
    train(trn_dataloader)