
'''
annotations에서 segmentation 제거
'''
import random
import json
import os
import chardet

from copy import deepcopy
from pycocotools.coco import COCO
from collections import defaultdict

coco_json_path = 'd:/TIPS dataset/export/merged_dataset_coco/annotations/3_clean.json'
split_rate = 0.8


with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

anns = coco_json['annotations']
new_anns = []
for ann in anns:
    if ann['segmentation']:
        continue
    else:
        new_anns.append(ann)
coco_json['annotations'] = new_anns

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)