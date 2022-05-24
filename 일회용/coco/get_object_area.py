'''
coco json 파일로부터 area 정보를 csv 파일로 출력
'''

import pandas as pd

from pycocotools.coco import COCO
from pathlib import Path

json_path = 'e:/Crater/불발탄및폭파구_area 계산/annotations/instances_default_filtered.json'
coco = COCO(json_path)

data = {'cat':[], 'area':[]}
for anno in coco.anns.values():
    catId = anno['category_id']
    data['cat'].append(coco.cats[catId]['name'])
    data['area'].append(anno['area'])

df_perObj = pd.DataFrame(data)
df_perCls = df_perObj.groupby('cat').mean()
with pd.ExcelWriter(Path(json_path).with_name('area.xlsx')) as f:
    df_perObj.to_excel(f,'perObj')
    df_perCls.to_excel(f,'perCls')

