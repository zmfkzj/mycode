'''
GT와 GT의 seg를 비교하여 IoU 산출
'''
from numpy.core.fromnumeric import mean
from pycocotools.coco import COCO
from pycocotools.mask import iou, frPyObjects
from operator import itemgetter

import numpy as np
import pandas as pd

gt = COCO('d:/compare GT/task_ai바우처 보고용 보현산댐 100장 gt(추후 삭제)-2021_11_04_15_24_10-coco 1.0/annotations/instances_default_30.json')
# comp_gt = COCO('d:/compare GT/task_ai바우처 보고용 보현산댐 하위 30장 라벨링_1-2021_11_04_15_54_26-coco 1.0/annotations/instances_default.json')
comp_gt = COCO('d:/compare GT/task_ai바우처 보고용 보현산댐 하위 30장 라벨링_2-2021_11_04_15_53_28-coco 1.0/annotations/instances_default.json')

# gt = COCO('d:/compare GT/task_ai바우처 보고용 보현산댐 100장 gt(추후 삭제)-2021_11_04_15_24_10-coco 1.0/annotations/instances_default_100.json')
# comp_gt = COCO('d:/compare GT/task_ai바우처 보고용 보현산댐 100장 gt_수정test-2021_11_04_17_25_05-coco 1.0/annotations/instances_default.json')

def eps(x,eps=1):
    return -eps*x+eps

comp_gt_imgs = comp_gt.imgs

datas = {'anns_id':[],'ious':[]}
for gt_img in gt.imgs.values():
    filename_getter = itemgetter('file_name')
    gt_img_filename = gt_img['file_name']
    gt_img_id = gt_img['id']
    comp_gt_img_id = [img['id'] for img in comp_gt_imgs.values() if filename_getter(img)==gt_img_filename ][0]

    height = gt_img['height']
    width = gt_img['width']

    gt_ann_ids = gt.getAnnIds(imgIds=gt_img_id)
    comp_gt_ann_ids = comp_gt.getAnnIds(imgIds=comp_gt_img_id)
    for gt_ann in gt.loadAnns(gt_ann_ids):
        iou_with_comp_gt=0
        for comp_gt_ann in comp_gt.loadAnns(comp_gt_ann_ids):
            gt_rle = frPyObjects(gt_ann['segmentation'],height,width)
            compare_gt_rle = frPyObjects(comp_gt_ann['segmentation'],height,width)
            IOU = iou(gt_rle,compare_gt_rle,np.zeros(1))
            # IOU = IOU + eps(np.mean([gt_ann['area'], comp_gt_ann['area']])/(height*width))*(1-IOU)
            if iou_with_comp_gt<IOU:
                iou_with_comp_gt = IOU.squeeze()
        datas['ious'].append(iou_with_comp_gt)
        datas['anns_id'].append(gt_ann['id'])


df = pd.DataFrame(datas)
df.to_csv('compare_result.csv')
