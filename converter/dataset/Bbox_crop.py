from pycocotools.coco import COCO
from PIL import Image

import cv2
import numpy as np
from pathlib import Path
from typing import List
import os
from collections import defaultdict

root = Path('d:/TIPS dataset/export/merged_dataset_include_rdd')
coco = COCO(str(root/'annotations/test.json'))
image_dir = root/'images'

def crop_bbox(img:np.ndarray, bbox:List[float]):
    x1,y1,w,h = [int(x) for x in bbox]
    crop_img = img[y1:y1+h,x1:x1+w,:]
    return crop_img

save_img_counts = defaultdict(int)

for imgId, img_dict in coco.imgs.items():
    img_filename = img_dict['file_name']
    img = Image.open(image_dir/img_filename).convert('RGB')
    img = np.array(img)

    annoIds = coco.getAnnIds(imgIds=imgId)
    annos = coco.loadAnns(annoIds)
    for anno in annos:
        bbox = anno['bbox']

        catId = anno['category_id']
        cat_name = coco.cats[catId]['name']
        crop_img = crop_bbox(img, bbox)

        if crop_img.size != 0:
            result, crop_img = cv2.imencode('.jpg',crop_img)
            if result:
                save_dir = root/f'crop_image/{cat_name}'
                if not os.path.isdir(str(save_dir)):
                    os.makedirs(str(save_dir),exist_ok=True)
                save_name =save_dir/f'test_{save_img_counts[cat_name]}.jpg'
                save_img_counts[cat_name] +=1
                with open(save_name, 'w+b') as f:
                    crop_img.tofile(f)

