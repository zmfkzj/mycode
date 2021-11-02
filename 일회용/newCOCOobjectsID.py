'''
voc to coco 후 annotations id가 1234,123식으로 엉클어진 것을 새로 만들어 줌
'''
import json
from pathlib import Path

coco_json_path = 'd:/pano_crop/data/annotations/instances_val.json'

with open(coco_json_path, 'r') as f:
    coco_json = json.load(f)

id = 0
for obj in coco_json['annotations']:
    obj['id'] = id
    id += 1

with open(coco_json_path,'w') as f:
    json.dump(coco_json,f)