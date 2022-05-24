'''
coco json에서 임의로 n개 이미지만 남겨놓음
'''
import json
import chardet
import random as rd

coco_json_path = 'd:/TIPS dataset/export/merged_dataset_exclude_rdd/annotations/instances_default.json'

with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

def pick_n_images(coco_json, n):
    images = coco_json['images'][:]
    rd.shuffle(images)
    if n>len(images):
        n = len(images)
    return images[:n]

coco_json['images'] = pick_n_images(coco_json,1000)

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)