'''
하나의 coco dataset을 train, test로 분리
각 class annotation 수 고려
'''
import random
import json
import os
import chardet

from copy import deepcopy
from pycocotools.coco import COCO
from collections import defaultdict

coco_json_path = 'j:/62Nas/mk/merged_dataset_coco/annotations/5_reclean.json'
split_rate = 0.8


with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

coco = COCO()
coco.dataset = coco_json
coco.createIndex()

cls_cnt = defaultdict(int)
for anno in coco.anns.values():
    cls_cnt[anno['category_id']]+=1

split_cnt = {catId:cnt*split_rate for catId, cnt in cls_cnt.items()}

train_imgs = []
train_annos = []
test_imgs = []
test_annos = []
current_cls_cnt = defaultdict(int)
img_anns = list(coco.imgToAnns.items())
random.shuffle(img_anns)
for imgId, annos in img_anns:
    cond = []
    for anno in annos:
        annoId = anno['id']
        annoCatId = anno['category_id']
        cond.append(split_cnt[annoCatId]>current_cls_cnt[annoCatId])
    if all(cond):
        train_imgs.append(coco.imgs[imgId])
        train_annos.extend(annos)
        for anno in annos:
            annoCatId = anno['category_id']
            current_cls_cnt[annoCatId] +=1
    else:
        test_imgs.append(coco.imgs[imgId])
        test_annos.extend(annos)

#train dataset
train_coco_json = deepcopy(coco.dataset)
train_coco_json['images'] = train_imgs
train_coco_json['annotations'] = train_annos


#test dataset
test_coco_json = deepcopy(coco.dataset)
test_coco_json['images'] = test_imgs
test_coco_json['annotations'] = test_annos

dir = os.path.split(coco_json_path)[0]
with open(os.path.join(dir,'train.json'),'w') as f:
    json.dump(train_coco_json,f)

with open(os.path.join(dir,'test.json'),'w') as f:
    json.dump(test_coco_json,f)