"""
Data_helper
Competition url: http://www.datafountain.cn/#/competitions/276/intro
"""

# -*- coding: utf-8 -*-
# @Time    : 2017/10/13 下午9:54
# @Author  : Qi MO
# @File    : data_helper.py
# @Software: PyCharm

import csv
import json
import os.path


class ArticleSample(object):
    def __init__(self, _id, title, deal_title, content, deal_content, judge,
                 deal_judge, cv):
        """
        文章对象
        :param id:  文章ID(str)
        :param title:  文章标题(str)
        :param deal_title: 处理后的文章标题(list)
        :param content:  文章内容(str)
        :param deal_content:  处理后的文章内容(list)
        :param judge:  结果(str: NEGATIVE或POSITIVE)
        :param deal_judge: 结果(1表示POSITIVE，0表示NEGATIVE)
        :param cv: 交叉验证标记(int)
        """
        self.id = _id
        self.title = title
        self.content = content
        self.judge = judge
        self.cv = cv
        self.deal_title = deal_title
        self.deal_content = deal_content
        self.deal_judge = deal_judge

    def __str__(self):
        info = {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'judege': self.judge,
            'cv': self.cv,
            'deal_title': self.deal_title,
            'deal_content': self.deal_content,
            'deal_judge': self.deal_judge
        }
        return json.dumps(info, indent=4)


def build_vocab(file_path, voc_path):
    """
    建立词典
    :param file_path: 建立词典文件地址
    :param voc_path: 词典保存地址
    :return:
    """
    print('build vocab...')
    voc = {}
    voc_index = 0
    max_title_length = 0
    max_content_length = 0
    with open(file_path, encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            title = line[1] if len(line) == 4 else ''
            content = line[-2]
            max_title_length = max(max_title_length, len(title))
            max_content_length = max(max_content_length, len(content))
            # if len(content) == 93749:
            #     print(title)
            #     print(content)
            for x in title:
                if x not in voc:
                    voc_index += 1
                    voc[x] = voc_index
            for x in content:
                if x not in voc:
                    voc_index += 1
                    voc[x] = voc_index
    print('build vocab done')
    voc_dict = {
        'voc': voc,
        'max_title_length': max_title_length,
        'max_content_length': max_content_length
    }
    with open(voc_path, 'w') as f:
        json.dump(voc_dict, f)
    return [voc, max_title_length, max_content_length]


def load_data_cv(file_path, voc_path, mode, cv=5):
    """
    加载训练数据、测试数据
    :param file_path:  数据地址
    :param voc_path:  词典地址（建立词典之后直接可以加载）
    :param mode:  是训练数据(train)还是测试数据(test)
    :param cv:  几折交叉验证
    :return:
    """
    rev = []
    if os.path.isfile(voc_path):
        with open(voc_path, 'r') as f:
            voc_dict = json.load(f)
            voc = voc_dict['voc']
            max_title_length = voc_dict['max_title_length']
            max_content_length = voc_dict['max_content_length']
    else:
        voc, max_title_length, max_content_length = build_vocab(
            file_path, voc_path)
    max_content_length = min(max_content_length, 1000)
    print('len voc: ', len(voc))
    print('max_title_length: ', max_title_length)
    print('max_content_length: ', max_content_length)

    print('load data...')
    cnt = 0
    with open(file_path, encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            cnt += 1
            if cnt % 20000 == 0:
                print('load data:...', cnt)
            if mode == 'train':
                _id = line[0]
                title = line[1] if len(line) == 4 else ''
                content = line[-2]
                judge = line[-1]
            else:
                _id = line[0]
                title = line[1] if len(line) == 3 else ''
                content = line[-1]
                judge = ''

            title, content = title[:
                                   max_title_length], content[:
                                                              max_content_length]
            deal_title = [voc[x] if x in voc else 0 for x in title]
            deal_title.extend([0] * (max_title_length - len(deal_title)))
            deal_content = [voc[x] if x in voc else 0 for x in content]
            deal_content.extend([0] * (max_content_length - len(deal_content)))
            deal_judge = [1, 0] if judge == 'POSITIVE' else [0, 1]
            article = ArticleSample(
                _id=_id,
                title=title,
                deal_title=deal_title,
                content=content,
                deal_content=deal_content,
                judge=judge,
                deal_judge=deal_judge,
                cv=cv)
            rev.append(article)

    print('len rev: ', len(rev))
    # print('rev 0...')
    # print(rev[0].title)
    # print(rev[0].deal_title)
    # print(rev[0].content)
    # print(rev[0].deal_content)
    # print(rev[0].judge)
    # print(rev[0].deal_judge)
    return rev, voc


if __name__ == '__main__':
    file_path_train = '../../docs/data/train.tsv'
    file_path_test = '../../docs/data/evaluation.tsv'
    voc_path = '../../docs/data/voc.json'
    artiles, voc = load_data_cv(file_path_train, voc_path, 'train')
    print(artiles[0], len(voc))
