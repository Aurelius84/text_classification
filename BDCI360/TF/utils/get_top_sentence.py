import warnings
import pandas as pd
import numpy as np
import time
warnings.filterwarnings("ignore")
import re
def go_split(s,min_len,symbol):
    # 拼接正则表达式
    #symbol = ';./+，；。！、'
    symbol = "[" + symbol + "]+"
    result = re.split(symbol, s)
    return [x for x in result if len(x)>min_len]
    #return result
def find_max(x,top=5):
    str = ''
    for item in sorted(x, key=lambda x: len(x))[-top:]:
        str=str+','+item
    return str.strip()




def make_tops(file,nrows = 100,top=5):
    start = time.clock()
    data = pd.read_csv(file,sep='\t',header=None,nrows = nrows,encoding='utf-8',names=['id','title','content','label'])
    #data['label'] = data['label'].apply(lambda x:int(x=='POSITIVE'))#人写的是1 机器写的是0
    data['content_split']=data.content.apply(lambda x: go_split(str(x),min_len=0,symbol = '，；。！？!'))
    data['content_split_max'] = data['content_split'].apply(lambda x:find_max(x,top=top) if len(x)>0 else '')
    data[['id','title','content_split_max','label']].to_csv(file+'_top_{0}_tsv'.format(top),sep='\t',header=None,index=False)
    print('done in %.2f s '%(time.clock()-start))


trainfile = 'data/train.tsv'


#在data目录下生成对应的文件 #file+'_top_{0}_tsv'
#top :选择最长的句子的个数
#nows:读取文件的行数

#最后的文件格式：
#id ,title ,top_sentences,label
make_tops(trainfile,nrows = 600000,top=5)
print('train flie got')
testfile = 'data/evaluation_public.tsv'
make_tops(testfile,nrows = 400000,top=5)
print('test flie got')