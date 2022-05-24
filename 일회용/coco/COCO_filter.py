'''
COCO json과 이미지를 비교, 이미지가 없으면 json에서 삭제
'''

import json
import os

from pycocotools.coco import COCO
from pathlib import Path

dir_path = Path('e:/Crater/새 폴더/211009_불발탄및폭파구_공간정보/')
json_path = dir_path/'annotations/instances_default.json'
images_path = dir_path/'images'

images = []
for r,ds,fs in os.walk(images_path):
    img_files = [Path(r)/f for f in fs if Path(f).suffix.lower() in ['.jpg', '.png']]
    images.extend(img_files)

coco = COCO(str(json_path))
coco_imgs_filename = [img['file_name'] for img in coco.imgs.values()]
img_filename_in_folder = [img_in_folder.name for img_in_folder in images]
filtered_coco_imgs = [img for img in coco.imgs.values() if Path(img['file_name']).name in img_filename_in_folder]

filtered_coco_img_ids = [img['id'] for img in filtered_coco_imgs]
filtered_coco_annos = [anno for anno in coco.anns.values() if anno['image_id'] in filtered_coco_img_ids]

coco.dataset['images'] = filtered_coco_imgs
coco.dataset['annotations'] = filtered_coco_annos

coco.createIndex()
coco.dataset
with open(json_path.with_name('instances_default_filtered.json'), 'w', encoding='cp949') as f:
    json.dump(coco.dataset,f)

