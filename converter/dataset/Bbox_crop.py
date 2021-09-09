from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
from typing import List
import os
from collections import defaultdict

root = Path('d:/pano16_coco')
coco = COCO(str(root/'annotations/instances_default.json'))
image_dir = root/'images'

def crop_bbox(img:np.ndarray, bbox:List[float]):
    x1,y1,w,h = [int(x) for x in bbox]
    crop_img = img[y1:y1+h,x1:x1+w,:]
    return crop_img

save_img_counts = defaultdict(int)

for imgId, img_dict in coco.imgs.items():
    img_filename = img_dict['file_name']
    img = np.fromfile(image_dir/img_filename,dtype=np.uint8)
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)

    annoIds = coco.getAnnIds(imgIds=imgId)
    annos = coco.loadAnns(annoIds)
    for anno in annos:
        if anno['segmentation']:
            bbox = anno['bbox']
        else:
            continue

        catId = anno['category_id']
        cat_name = coco.cats[catId]['name']
        crop_img = crop_bbox(img, bbox)

        result, crop_img = cv2.imencode('.jpg',crop_img)
        if result:
            save_dir = root/f'crop_image/{cat_name}'
            if not os.path.isdir(str(save_dir)):
                os.makedirs(str(save_dir),exist_ok=True)
            save_name =save_dir/f'{save_img_counts[cat_name]}.jpg'
            save_img_counts[cat_name] +=1
            with open(save_name, 'w+b') as f:
                crop_img.tofile(f)

