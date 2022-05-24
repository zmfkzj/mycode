from pathlib import Path
import json
from collections import defaultdict
from functools import reduce

path = Path('C:/Users/mkkim/Desktop/RTK DATASET/RTK_SemanticSegmentationGT_Json')
json_files = path.glob('*.json')

coco = {
            "licenses": [
                {
                    "name": "",
                    "id": 0,
                    "url": ""
                }
            ],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            'categories':[],
            'images':[],
            'annotations':[]}

categories = defaultdict(lambda: len(categories)+1)

for idx,file in enumerate(json_files):
    with open(file, 'r') as f:
        contents = json.load(f)
    image_data = {'id':idx+1, 
                'width': contents['imageHeight'],
                "height": contents['imageWidth'],
                "file_name": str(file.with_suffix('.png').name),
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0}
    coco["images"].append(image_data)
    for shape in contents['shapes']:
        anno_data = {
                    "id": len(coco["annotations"])+1,
                    "image_id": idx+1,
                    "category_id": categories[shape['label']],
                    "segmentation": [list(reduce(lambda x,y: x+y,shape['points']))],
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False
                    },
                    "area": 22832,
                    "bbox": [
                        385.81,
                        198.07,
                        226.99,
                        211.81
                    ]
                }
        coco["annotations"].append(anno_data)


for cat,id in categories.items():
    cat_data = {
            "id": id,
            "name": cat,
            "supercategory": ""
            }
    coco["categories"].append(cat_data)

with open('RTK.json', 'w') as f:
    json.dump(coco,f)