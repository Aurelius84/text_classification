#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/25 下午2:06
# @From    : PyCharm
# @File    : BaseModule.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import torch
import torch.nn as nn
import time

class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

        self.model_name = str(type(self))

    def save(self, name=None, dir=None):
        '''
        save model to path
        Args:
            name: str, model name,default(className)
            new:

        Returns: str, path of model

        '''
        dir = dir or '../checkpoints/'
        prefix = dir + self.model_name + '_' + self.opt.input_type + '_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name

        data = {'opt': self.opt, 'd': self.state_dict()}

        torch.save(data, path)

        return path

    def load(self, path, change_opt=True):
        '''
        Load model from path
        Args:
            path: str, saved path of model file
            change_opt: boolean

        Returns: model

        '''
        print(path)
        data = torch.load(path)
        if 'opt' in data:
            if change_opt:
                self.opt.embed_path = None
                self.__init__(self.opt)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def get_optimizer(self, lr1, lr2=None, weight_decay=0):
        '''

        Args:
            lr1:
            lr2:
            weight_decay:

        Returns:

        '''
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        if lr2 is None: lr2 = lr1 * 0.5
        optimizer = torch.optim.RMSprop([
            dict(params=base_params, weight_decay=weight_decay, lr=lr1),
            {'params': self.encoder.parameters(), 'lr': lr2}
        ])
        return optimizer
