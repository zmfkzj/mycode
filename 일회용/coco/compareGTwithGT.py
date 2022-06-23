'''
GT와 GT의 seg, or bbox를 비교하여 IoU 산출
'''
from numpy.core.fromnumeric import mean
from pycocotools.coco import COCO
from pycocotools.mask import iou, frPyObjects
from operator import itemgetter

import numpy as np
import pandas as pd
import json
import chardet

gt_path = 'j:/62Nas/mk/merged_dataset_coco/annotations/coco_results.json'
comp_gt_path = 'j:/62Nas/mk/merged_dataset_coco/annotations/test_2.json'

iou_type = 'bbox'




with open(gt_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(gt_path, 'r',encoding=encoding) as f:
    gt_json = json.load(f)

with open(comp_gt_path, 'r+b') as f:
    encoding = chardet.detect(f.read())['encoding']
with open(comp_gt_path, 'r',encoding=encoding) as f:
    comp_gt_json = json.load(f)

gt = COCO()
comp_gt = COCO()

gt.dataset = gt_json
comp_gt.dataset = comp_gt_json

gt.createIndex()
comp_gt.createIndex()

def eps(x,eps=1):
    return -eps*x+eps

def bbox_to_segm(bbox):
    x1,y1,w,h = bbox
    x2 = x1+w
    y2 = y1+h
    return [[x1,y1,x2,y1,x2,y2,x1,y2]]


comp_gt_imgs = comp_gt.imgs

datas = {'gt_image_id':[],'gt_anns_id':[],'comp_gt_anns_id':[],'ious':[]}
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
        max_IOU_comp_gt_id = -1
        for comp_gt_ann in comp_gt.loadAnns(comp_gt_ann_ids):
            if iou_type=='bbox':
                IOU = iou([gt_ann['bbox']],[comp_gt_ann['bbox']],np.zeros(1))
            elif iou_type=='segm':
                gt_rle = gt.annToRLE(gt_ann)
                compare_gt_rle = comp_gt.annToRLE(comp_gt_ann)
                IOU = iou(gt_rle,compare_gt_rle,np.zeros(1))
            # IOU = IOU + eps(np.mean([gt_ann['area'], comp_gt_ann['area']])/(height*width))*(1-IOU)
            if iou_with_comp_gt<IOU:
                iou_with_comp_gt = IOU.squeeze()
                max_IOU_comp_gt_id = comp_gt_ann['id']
        datas['ious'].append(iou_with_comp_gt)
        datas['gt_anns_id'].append(gt_ann['id'])
        datas['gt_image_id'].append(gt_ann['image_id'])
        datas['comp_gt_anns_id'].append(max_IOU_comp_gt_id)


df = pd.DataFrame(datas)
df.to_csv('compare_result.csv')
