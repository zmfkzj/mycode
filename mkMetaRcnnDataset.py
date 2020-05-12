from predict_result import process_gt
import pandas as pd
from util.filecontrol import pickFilename
from os.path import *


def mkdataset(filelist, subset, root, form='voc', per='class'):
    gtpart = process_gt(filelist, subset, root, form=form)
    uniq = gtpart[per].unique()
    gtpart.set_index(per, inplace=True)
    for val in uniq:
        dataset = gtpart.loc[val, ['img']].map(pickFilename)
        dataset.to_csv(join(root, f'ImageSets/Main/{val}_{subset}.txt'), encoding='euc-kr', index=False, header=False)

if __name__ == "__main__":
    root = '/home/tm/nasrw/mk/MetaR-CNN/dataset/vocform'
    filelist = join(root, 'ImageSets/Main/default.txt')
    subset = pickFilename(filelist)

    mkdataset(filelist, subset, root, form='voc', per='class')

    