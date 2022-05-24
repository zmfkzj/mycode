'''
coco json의 categories name을 변경 
-> category name이 같으면 id 통합 
-> annotation category_id도 같이 변경
'''
import json
import chardet

from collections import defaultdict

coco_json_path = 'd:/TIPS dataset/export/merged_dataset_coco/annotations/2_chg_cat.json'

with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

#category name이 같으면 id 통합
dict_name_cat = defaultdict(list)
for cat in coco_json['categories']:
    dict_name_cat[cat['name']].append(cat)

new_cat = []
dict_oldCatId_newCatId = {}
for cat_name, cats in dict_name_cat.items():
    new_cat.append(cats[0])
    for cat in cats:
        dict_oldCatId_newCatId[cat['id']] = cats[0]['id']
coco_json['categories'] = new_cat

# annotation category_id도 같이 변경
for idx, anno in enumerate(coco_json['annotations']):
    try:
        coco_json['annotations'][idx]['category_id'] = dict_oldCatId_newCatId[anno['category_id']]
    except KeyError:
        pass

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)