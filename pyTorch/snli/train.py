#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/14 下午8:40
# @From    : PyCharm
# @File    : train.py
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import os
import time
import glob

import torch
import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier
from util import get_args, makedirs

args = get_args()

torch.cuda.set_device(args.gpu)
inputs = data.Field(lower=args.lower)
answer = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answer)

inputs.build_vocab(train, dev, test)

if args.word_vectors:
    if os.path.isfile(args.word_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)

answer.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, device=args.gpu
)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answer.vocab)
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))

else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        model.train()
        optimizer.zero_grad()
        iterations += 1

        answer = model(batch)

        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total

        loss = criterion(answer, batch.label)

        loss.backward()
        optimizer.step()

        if iterations % args.save_every == args.save_every - 1:
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                 if f != snapshot_path:
                     os.remove(f)

        if iterations % args.dev_every == args.dev_every - 1:
            model.eval()
            dev_iter.init_epoch()

            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += (
                torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0],
                                          train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc,
                                                                                                   dev_loss.data[0],
                                                                                                   iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
        elif iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      n_correct / n_total * 100, ' ' * 12))
