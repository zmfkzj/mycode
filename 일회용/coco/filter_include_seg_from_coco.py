'''
coco dataset에서 seg가 있는 annotation만 남겨놓음
'''
import json
import chardet

coco_json_path = 'd:/TIPS dataset/export/merged_dataset_coco/annotations/instances_default.json'

with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

new_annotations = []
for obj in coco_json['annotations']:
    if obj['segmentation']:
        new_annotations.append(obj)

coco_json['annotations'] = new_annotations

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)