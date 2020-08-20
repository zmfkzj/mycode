import pandas as pd
import os
from glob import glob
import re
from tqdm import tqdm
from collections import defaultdict
'''
실험 결과 데이터 모으기
'''
defect_work_path = os.path.expanduser('~/nasrw/mk/work_dataset')
defect_work_dirs = glob(os.path.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_*'))

casenum_dict = pd.read_csv(os.path.expanduser('~/nasrw/mk/work_dataset/casenum.csv')).set_index(['weight', 'csv']).to_dict()['case']
casenum_dict = defaultdict(int,casenum_dict)

cols = ['case','부식','백태','박리','철근노출','파손','박락','누수','재료분리','들뜸','total']
perDataset_files = []
for dir in defect_work_dirs:
    for p, ds, fs in os.walk(dir):
        if fs:
            for f in fs:
                if re.match('.*_perDataset_.*csv', f):
                    perDataset_files.append(os.path.join(p,f))

recall = pd.DataFrame()
precision = pd.DataFrame()
F1 = pd.DataFrame()

for perDataset_path in perDataset_files:
    perDataset_df = pd.read_csv(perDataset_path, encoding='euc-kr')
    weight = list(filter(lambda x: re.match('2DOD_defect_.*', x)!=None , perDataset_path.split('/')))[0]
    csvfile = os.path.basename(perDataset_path)
    if not bool(weight):
        raise

    casenum = casenum_dict[(weight, csvfile)]
    _recall = perDataset_df['recall'].rename((weight, csvfile))*100
    _recall.index = perDataset_df['class']
    _recall['case'] = casenum
    _precision = perDataset_df['precision'].rename((weight, csvfile))*100
    _precision.index = perDataset_df['class']
    _precision['case'] = casenum
    _F1 = perDataset_df['F1-Score'].rename((weight, csvfile))*100
    _F1.index = perDataset_df['class']
    _F1['case'] = casenum

    recall = recall.append(_recall)
    precision = precision.append(_precision)
    F1 = F1.append(_F1)

recall.index = pd.MultiIndex.from_tuples(recall.index.values)
precision.index = pd.MultiIndex.from_tuples(precision.index.values)
F1.index = pd.MultiIndex.from_tuples(F1.index.values)

pd.DataFrame(recall, columns=cols).fillna('-').to_csv(os.path.join(defect_work_path, 'recall.csv'), encoding='euc-kr')
pd.DataFrame(precision, columns=cols).fillna('-').to_csv(os.path.join(defect_work_path, 'precision.csv'), encoding='euc-kr')
pd.DataFrame(F1, columns=cols).fillna('-').to_csv(os.path.join(defect_work_path, 'F1.csv'), encoding='euc-kr')
