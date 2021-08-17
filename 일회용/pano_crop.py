import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Polygon,PolygonsOnImage


import json
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO

coco_dataset = Path('d:/pano16_coco')
image_dir = coco_dataset/'images'

coco = COCO(str(coco_dataset/'annotations/instances_default.json'))


ia.seed(1)
def cvtSeg(coco_seg):
    return [(coco_seg[idx],coco_seg[idx+1]) for idx in range(0,len(coco_seg),2)]

def crop(image, polygons):
    polys = PolygonsOnImage(polygons,shape=image.shape)

    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=1000, height=600),
        iaa.ClipCBAsToImagePlanes()
    ])

    image_aug, polys_aug = seq(image=image, polygons=polys)
    return image_aug, polys_aug

for image_id in coco.getImgIds():
    coco_image = coco.loadImgs(image_id)[0]

    image = np.fromfile(image_dir/coco_image['file_name'],np.uint8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)

    coco_annos = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    coco_labels = [coco.loadCats(coco.getCatIds(catIds=anno['category_id'])[0])[0]['name'] for anno in coco_annos]
    polygons = [Polygon(cvtSeg(anno['segmentation'][0]),label=1) for anno in coco_annos]
    crop_image, crop_anno = crop(image, polygons)
    print()
