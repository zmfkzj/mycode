'''
voc to coco 후 annotations id가 1234,123식으로 엉클어진 것을 새로 만들어 줌
'''
import json
import chardet

coco_json_path = 'd:/pano_crop/SQ_NEU_Dam_coco/annotations/instances_test.json'

with open(coco_json_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(coco_json_path, 'r',encoding=encoding) as f:
    coco_json = json.load(f)

id = 1
for obj in coco_json['annotations']:
    obj['id'] = id
    id += 1

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)