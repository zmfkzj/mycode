import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.augmentables.batches import UnnormalizedBatch
import os
import os.path as osp
import imageio
import cv2
import numpy as np
np.random.bit_generator = np.random._bit_generator

from PIL import Image
import aug_cfg
from util.filecontrol import *
from converter.annotation.voc2yolo import calxyWH, load_annotation
from converter.annotation.yolo2voc import calLTRB, create_file
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
from itertools import repeat
from multiprocessing import Pool

# ia.seed(1)
Image.MAX_IMAGE_PIXELS = 1000000000

############################################################################
#configure

BATCH_SIZE:int = 10
goal:int = 10
inputpath = "/home/tm/nasrw/결함-YOLOv3/cvat upload/task_결함-2020_04_06_05_28_07-yolo/obj_train_data"
suffix = ''
file_preffix = ''
# seqs = [aug_cfg.seq_Crop, aug_cfg.seq_Pad]
aug_option = [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.geometric.Rot90((0,3),keep_size=False),
        iaa.Multiply((0.95,1.05), per_channel=True),
        iaa.MultiplyHueAndSaturation((0.5,1.5),per_channel=False),
        iaa.MotionBlur(k=(3,5)),
        iaa.Grayscale((0.0,1.0)),
        iaa.RemoveCBAsByOutOfImageFraction(0.5),
        iaa.ClipCBAsToImagePlanes()
        ]
augmentation = 0
drawbbox = 1
drawlabel = True
clip = False
# downscale = (1400,1400) # tuple, list or None
downscale = None
multiproc = 0
bbox_color = aug_cfg.bbox_color
noob_save = 0 #no object image save

############################################################################
#functions

def mkfileDict(root:str, extlist:Union[list, tuple]=['.png', '.jpg', '.txt', '.xml']) -> dict:
    '''
    폴더 및 하위 폴더의 확장자가 extlist 안에 있는 모든 파일을 불러와 dict에 분류
    return files = {
        'filename' :{
            'ext' : filepath
        }
    }
    '''
    filepathlist:list = folder2list(root,extlist)
    files = defaultdict(lambda: defaultdict(str))
    for file in filepathlist:
        basename = osp.basename(file)
        filename, ext = osp.splitext(basename)
        files[filename][ext] = file
    return dict(files)

def Fileclassify(filedict:Dict[str, Dict[str, str]]) -> dict:
    labeldict = defaultdict(lambda: defaultdict(str))
    for filename, val in filedict.items():
        try:
            labeldict[filename]['bbox'] = dict(val)['.xml']
        except KeyError:
            try:
                labeldict[filename]['bbox'] = dict(val)['.txt']
            except KeyError:
                pass

        ispng = '.png' in val.keys()
        isjpg = '.jpg' in val.keys()
        if isjpg & ispng:
            labeldict[filename]['img'] = val['.jpg']
            labeldict[filename]['mask'] = val['.png']
        elif isjpg ^ ispng:
            if isjpg:
                labeldict[filename]['img'] = val['.jpg']
            elif ispng:
                labeldict[filename]['img'] = val['.png']
        else:
            print(f'{val.values()} 파일과 매칭되는 이미지 파일이 존재하지 않습니다.')
    labeldict = {key:val for key, val in labeldict.items() if bool(val['img'])&(bool(val['bbox'])|bool(val['mask']))}
    return dict(labeldict)

def load_aug(labeldict:dict, root):
    image = imageio.imread(osp.join(root, labeldict['img']))
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    root_basename = os.path.basename(root)
    newpath = {attr: osp.join(root,f'..', f'{root_basename}_aug', val) for attr, val in labeldict.items()}

    shape = image.shape
    if labeldict['bbox'].endswith('txt'):
        bboxeslist = mkbboxlist(osp.join(root, labeldict['bbox']), shape)
    else:
        bboxeslist = mkbboxlist(osp.join(root, labeldict['bbox']))
    bbs = BoundingBoxesOnImage(bboxeslist, shape=shape)

    mask = None
    if labeldict['mask']:
        mask = SegmentationMapOnImage(imageio.imread(osp.join(root, labeldict['mask'])), shape)
    
    return image, bbs, mask, newpath

def save_aug(aug, idx, newpath:dict):
    aug_img = aug.images_aug[idx]
    shape = aug_img.shape
    aug_bboxs = aug.bounding_boxes_aug[idx]
    aug_mask = None
    if aug.segmentation_maps_aug[idx]:
        aug_mask = aug.segmentation_maps_aug[idx]

    if aug_bboxs:
        newfilepath = {attr: addsuffix(val, str(c)) for attr, val in newpath.items()}
        os.makedirs(osp.split(newfilepath['img'])[0], exist_ok=True)
        imageio.imwrite(newfilepath['img'], aug_img)
        if newfilepath['bbox'].endswith('txt'):
            xywh_boxes = []
            for bbox in aug_bboxs:
                xywh = calxyWH(bbox.coords, shape)
                xywh_boxes.append([bbox.label, *xywh])
            os.makedirs(osp.split(newfilepath['bbox'])[0], exist_ok=True)
            list2txt(xywh_boxes, newfilepath['bbox'])
        elif newfilepath['bbox'].endswith('xml'):
            ltrb_boxes = []
            for bbox in aug_bboxs:
                ltrb_boxes.append([bbox.label, bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int])
            xml_tree = create_file(newfilepath['img'], shape[1],shape[0], ltrb_boxes)
            os.makedirs(osp.split(newfilepath['bbox'])[0], exist_ok=True)
            xml_tree.write(newfilepath['bbox'])
        if aug_mask:
            os.makedirs(osp.split(newfilepath['mask'])[0], exist_ok=True)
            imageio.imwrite(newfilepath['mask'], aug_mask)

def run_aug(images:np.ndarray, bbs, mask, newpath:Dict[str,str]) -> List[str]:
    c = 0
    dataset:List[str] = []
    while c < goal:
        batche = UnnormalizedBatch(images=list(repeat(images, BATCH_SIZE)), 
                                   bounding_boxes=list(repeat(bbs, BATCH_SIZE)), 
                                   segmentation_maps=list(repeat(mask, BATCH_SIZE)))
        seq = iaa.Sequential(aug_option)
        aug = list(seq.augment_batches(batche, background=True))[0]
        for idx in range(BATCH_SIZE):
            aug_img = aug.images_aug[idx]
            shape = aug_img.shape
            aug_bboxs = aug.bounding_boxes_aug[idx]
            aug_mask = None
            if aug.segmentation_maps_aug:
                aug_mask = aug.segmentation_maps_aug[idx]

            if aug_bboxs:
                newfilepath = {attr: addsuffix(val, str(c+idx)) for attr, val in newpath.items()}
                os.makedirs(osp.split(newfilepath['img'])[0], exist_ok=True)
                imageio.imwrite(newfilepath['img'], aug_img)
                if newfilepath['bbox'].endswith('txt'):
                    xywh_boxes = []
                    for bbox in aug_bboxs:
                        xywh = calxyWH(bbox.coords, shape)
                        xywh_boxes.append([bbox.label, *xywh])
                    os.makedirs(osp.split(newfilepath['bbox'])[0], exist_ok=True)
                    list2txt(xywh_boxes, newfilepath['bbox'])
                elif newfilepath['bbox'].endswith('xml'):
                    ltrb_boxes = []
                    for bbox in aug_bboxs:
                        ltrb_boxes.append([bbox.label, bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int])
                    xml_tree = create_file(newfilepath['img'], shape[1],shape[0], ltrb_boxes)
                    os.makedirs(osp.split(newfilepath['bbox'])[0], exist_ok=True)
                    xml_tree.write(newfilepath['bbox'])
                if aug_mask:
                    os.makedirs(osp.split(newfilepath['mask'])[0], exist_ok=True)
                    imageio.imwrite(newfilepath['mask'], aug_mask)
                dataset.append(pickFilename(newfilepath['img']))
        c+=BATCH_SIZE
    return dataset

sizetype = Optional[Union[Tuple[int, int], List[int]]]
def mkbboxlist(labelpath:str, imgsize:sizetype=None):
    if labelpath.endswith('.txt'):
        assert bool(imgsize), 'imagesize를 입력해 주세요.'
        h, w = imgsize
        objs = [list(map(num, line.rstrip('\n').split(' '))) for line in txt2list(labelpath)]
        bboxes = [[int(obj[0]), *calLTRB(obj[1:], ).tolist()] for obj in objs]
        bboxeslist:List[BoundingBox] = [BoundingBox(x1=bbox[1]*w,
                                                    x2=bbox[3]*w,
                                                    y1=bbox[2]*h,
                                                    y2=bbox[4]*h,
                                                    label=bbox[0]) for bbox in bboxes]
    else:
        anno = load_annotation(labelpath)
        bboxeslist:List[BoundingBox] = [BoundingBox(x1=bbox[1],
                                                    x2=bbox[3],
                                                    y1=bbox[2],
                                                    y2=bbox[4],
                                                    label=bbox[0]) for bbox in anno['obj']]

    return bboxeslist

def main(root:str):
    labeldict = Fileclassify(mkfileDict(root))
    # for idx in range(0, len(labeldict),BATCH_SIZE):
    #     augs = list(labeldict.values())[idx:idx+BATCH_SIZE]
    #     augs = list(map(lambda val: load_aug(val, root), augs))
    #     run_aug(*zip(*augs))

    dataset = []
    for val in tqdm(labeldict.values(), desc='image augmentation'):
        dataset.extend(run_aug(*load_aug(val, root)))
    list2txt(dataset, osp.join(root, 'default.txt'))


class InputFileFormatError(Exception):
    def __str__(self):
        return 'inputpath가 존재하는 폴더나 텍스트파일이 아닙니다.'

def rm_zerobbox(bbox_aug):
    xyxy = bbox_aug.to_xyxy_array()
    voc_boxes = []
    for i in range(xyxy.shape[0]):
        # voc_box = voc2yolo(xyxy[i,:], bbox_aug.shape)
        xshape = bbox_aug.shape[1]
        yshape = bbox_aug.shape[0]
        c_x = bbox_aug.bounding_boxes[i].center_x/xshape
        c_y = bbox_aug.bounding_boxes[i].center_y/yshape

        if (c_x<=0) | (c_y<=0) | (c_x>=1) | (c_y>=1):
            continue

        if bbox_aug.bounding_boxes[i].x1<0: bbox_aug.bounding_boxes[i].x1 = 0
        if bbox_aug.bounding_boxes[i].x1>xshape: bbox_aug.bounding_boxes[i].x1 = xshape
        if bbox_aug.bounding_boxes[i].x2<0: bbox_aug.bounding_boxes[i].x2 = 0
        if bbox_aug.bounding_boxes[i].x2>xshape: bbox_aug.bounding_boxes[i].x2 = xshape
        if bbox_aug.bounding_boxes[i].y1<0: bbox_aug.bounding_boxes[i].y1 = 0
        if bbox_aug.bounding_boxes[i].y1>yshape: bbox_aug.bounding_boxes[i].y1 = yshape
        if bbox_aug.bounding_boxes[i].y2<0: bbox_aug.bounding_boxes[i].y2 = 0
        if bbox_aug.bounding_boxes[i].y2>yshape: bbox_aug.bounding_boxes[i].y2 = yshape

        c_x = bbox_aug.bounding_boxes[i].center_x/xshape
        c_y = bbox_aug.bounding_boxes[i].center_y/yshape
        w = bbox_aug.bounding_boxes[i].width/xshape
        h = bbox_aug.bounding_boxes[i].height/yshape
        label =bbox_aug.bounding_boxes[i].label 
        
        voc_box = [label, c_x, c_y, w, h]
        voc_box = [str(ii) for ii in voc_box]
        voc_box = ' '.join(voc_box)
        voc_boxes.append(voc_box)
    return voc_boxes

######################################################################

if __name__ == "__main__":
    main('/home/tm/nasrw/mk/MetaR-CNN/dataset/VOC2007')