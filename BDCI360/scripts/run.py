# coding:utf-8

if __name__ == '__main__':
    new_train = '../docs/data/evaluation_public.tsv'
    old_train = '../docs/result/10_17.csv'
    new_save = '../docs/data/eval_add.tsv'
    old_save = '../docs/data/eval_old.tsv'
    with open(old_train, 'r', encoding='utf8') as fo, open(new_train, 'r', encoding='utf8') as fn, open(new_save, 'w', encoding='utf8') as fnw, open(old_save, 'w', encoding='utf8') as fow:
        old_ids = set([line.split(',')[0].strip() for line in fo])
        # print(old_ids)
        old_cnt, new_cnt, total_cnt = 0, 0, 0
        for data in fn:
            total_cnt += 1
            # print(data.split('\t')[0].strip())
            # exit()
            if data.split('\t')[0].strip() in old_ids:
                old_cnt += 1
                fow.write(data)
            else:
                new_cnt += 1
                fnw.write(data)

            if total_cnt % 1000 == 0:
                print('total: {}, new: {}, old: {}'.format(total_cnt, new_cnt, old_cnt))




