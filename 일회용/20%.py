import pandas as pd
import os.path as osp
import os
import numpy as np
import re
from glob import glob
from collections import defaultdict
'''
각 class에서 F1-Score 상위 20%의 이미지를 골라내어 다시 계산
'''

# path = osp.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_20200626_YOLOv2_1/train_perImg_yolov2-voc_6000_It0.5_ct0.25_200626-0904.csv')

defect_work_path = osp.expanduser('~/nasrw/mk/work_dataset')
defect_work_dirs = glob(osp.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_*'))

casenum_dict = pd.read_csv(os.path.expanduser('~/nasrw/mk/work_dataset/casenum_20%.csv')).set_index(['weight', 'csv']).to_dict()['case']
casenum_dict = defaultdict(int,casenum_dict)

cols = ['case','부식','백태','박리','철근노출','파손','박락','누수','재료분리','들뜸','total']

perImg_files = []
for dir in defect_work_dirs:
    for p, ds, fs in os.walk(dir):
        if fs:
            for f in fs:
                if re.match('.*_perImg.*csv', f):
                    perImg_files.append(osp.join(p,f))

recall = pd.DataFrame()
precision = pd.DataFrame()
F1 = pd.DataFrame()

for idx, path in enumerate(perImg_files):
    perImg = pd.read_csv(path, encoding='euc-kr').sort_values('F1-Score', ascending=False)

    weight = list(filter(lambda x: re.match('2DOD_defect_.*', x)!=None , path.split('/')))[0]
    csvfile = osp.basename(path)
    if not bool(weight):
        raise

    classes = perImg['class'].unique()
    
    perDataset = pd.DataFrame()

    for cls in classes:
        if cls == 'total':
            continue
        top20_count = int(np.around(len(perImg.loc[perImg['class']==cls,'img'].unique())*0.2))
        top20 = perImg.loc[perImg['class']==cls, ['FN_count', 'FP_count', 'TP_count', 'img', 'class']].iloc[:top20_count]

        perDataset = perDataset.append(top20, ignore_index=True)
    perDataset = perDataset.drop_duplicates('img')
    perDataset = perDataset.groupby('class').sum()
    perDataset = perDataset.append(perDataset.sum().rename('total'))
    perDataset['recall'] = perDataset['TP_count']/(perDataset['TP_count'] + perDataset['FN_count'])
    perDataset['precision'] = perDataset['TP_count']/(perDataset['TP_count'] + perDataset['FP_count'])
    perDataset['F1-Score'] = 2*perDataset['recall']*perDataset['precision']/(perDataset['recall']+perDataset['precision'])
    perDataset = perDataset.reindex(['부식', '백태', '박리', '철근노출', '파손', '박락', '누수', '재료분리', '들뜸', 'total'])

    casenum = casenum_dict[(weight, csvfile)]
    _recall = perDataset['recall'].rename((weight, csvfile))*100
    _recall.index = perDataset.index
    _recall['case'] = casenum
    _precision = perDataset['precision'].rename((weight, csvfile))*100
    _precision.index = perDataset.index
    _precision['case'] = casenum
    _F1 = perDataset['F1-Score'].rename((weight, csvfile))*100
    _F1.index = perDataset.index
    _F1['case'] = casenum

    recall = recall.append(_recall)
    precision = precision.append(_precision)
    F1 = F1.append(_F1)

recall.index = pd.MultiIndex.from_tuples(recall.index.values)
precision.index = pd.MultiIndex.from_tuples(precision.index.values)
F1.index = pd.MultiIndex.from_tuples(F1.index.values)

recall = pd.DataFrame(recall, columns=cols).fillna('-')
recall['mean'] = recall.mean(axis=1).rename('mean')
recall.to_csv(os.path.join(defect_work_path, 'recall_20%.csv'), encoding='euc-kr')

precision = pd.DataFrame(precision, columns=cols).fillna('-')
precision['mean'] = precision.mean(axis=1).rename('mean')
precision.to_csv(os.path.join(defect_work_path, 'precision_20%.csv'), encoding='euc-kr')

F1 = pd.DataFrame(F1, columns=cols).fillna('-')
F1['mean'] = F1.mean(axis=1).rename('mean')
F1.to_csv(os.path.join(defect_work_path, 'F1_20%.csv'), encoding='euc-kr')