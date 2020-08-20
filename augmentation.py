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
from itertools import repeat, chain
from multiprocessing import Pool
from toolz import curry
import asyncio as aio
import sys

# ia.seed(1)
Image.MAX_IMAGE_PIXELS = 1000000000

############################################################################
#configure

BATCH_SIZE:int = 5
nbBATCH = 3
goal:int = 15 # 목표 배수 10배면 10
# seqs = [aug_cfg.seq_Crop, aug_cfg.seq_Pad]
aug_option = [
        iaa.CropToFixedSize(width=1000, height=1000),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.geometric.Rot90((0,3),keep_size=False),
        iaa.Multiply((0.95,1.05), per_channel=True),
        iaa.MultiplyHueAndSaturation((0.5,1.5),per_channel=False),
        iaa.MotionBlur(k=(3,5)),
        iaa.Grayscale(1),
        iaa.RemoveCBAsByOutOfImageFraction(0.8),
        iaa.ClipCBAsToImagePlanes()
        ]
# downscale = (1400,1400) # tuple, list or None
noob_save = 0 #no object image save

############################################################################
#functions

def mkfileDict_fromfolder(root:str, extlist:Union[list, tuple]=['.png', '.jpg', '.txt', '.xml']) -> dict:
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

async def co_imwrite(path, src):
    print('save start', path)
    imageio.imwrite(path, src)
    print('save end', path)

@curry
async def save_aug(seq, batches, newpath):
    c = 0
    dataset = []
    while c<goal:

        print('agumentation start', c)
        augs = list(seq.augment_batches(batches, background=True))
        print('agumentation end', c)
        for aug in augs:
            for idx in range(len(aug.images_aug)):
                aug_img = aug.images_aug[idx]
                shape = aug_img.shape
                aug_bboxs = aug.bounding_boxes_aug[idx]
                aug_mask = None
                if aug.segmentation_maps_aug:
                    aug_mask = aug.segmentation_maps_aug[idx]

                if aug_bboxs:
                    print('process start')
                    newfilepath = {attr: addsuffix(val, str(c)) for attr, val in newpath.items()}
                    os.makedirs(osp.split(newfilepath['img'])[0], exist_ok=True)
                    dataset.append(pickFilename(newfilepath['img']))
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
                        await aio.create_task(co_imwrite(newfilepath['mask'], aug_mask))
                    await aio.create_task(co_imwrite(newfilepath['img'], aug_img))
                    c += 1
                    print('process end')

    return dataset

def run_aug(images:np.ndarray, bbs, mask, newpath:Dict[str,str]) -> List[str]:
    dataset:List[str] = []
    async def run_aug():
        c = 0
        tasks = []
        batches = [UnnormalizedBatch(images=list(repeat(images, BATCH_SIZE)), 
                                bounding_boxes=list(repeat(bbs, BATCH_SIZE)), 
                                segmentation_maps=list(repeat(mask, BATCH_SIZE))) for _ in range(nbBATCH)]
        seq = iaa.Sequential(aug_option)
        dataset = await save_aug(seq, batches, newpath)
        return dataset

    dataset = aio.run(run_aug())
    return dataset

sizetype = Optional[Union[Tuple[int, int], List[int]]]
def mkbboxlist(labelpath:str, imgsize:sizetype=None):
    if labelpath.endswith('.txt'):
        assert bool(imgsize), 'imagesize를 입력해 주세요.'
        h, w = imgsize
        objs = [list(map(num, line.rstrip('\n').split(' '))) for line in txt2list(labelpath)]
        bboxes = [[int(obj[0]), *calLTRB(obj[1:], ).tolist()] for obj in objs]
        bboxes = [[bbox[0], bbox[1]*w, bbox[2]*h, bbox[3]*w, bbox[4]*h] for bbox in bboxes]
        # bboxeslist:List[BoundingBox] = [BoundingBox(x1=bbox[1]*w,
        #                                             x2=bbox[3]*w,
        #                                             y1=bbox[2]*h,
        #                                             y2=bbox[4]*h,
        #                                             label=bbox[0]) for bbox in bboxes]
    else:
        anno = load_annotation(labelpath)
        bboxes = anno['obj']

    bboxeslist:List[BoundingBox] = [BoundingBox(x1=bbox[1],
                                                x2=bbox[3],
                                                y1=bbox[2],
                                                y2=bbox[4],
                                                label=bbox[0]) for bbox in bboxes]

    return bboxeslist

def main(root:str):
    labeldict = Fileclassify(mkfileDict_fromfolder(root))

    dataset = []
    for val in tqdm(labeldict.values(), desc='image augmentation'):
        dataset.extend(run_aug(*load_aug(val, root)))
    print(dataset)
    list2txt(dataset, osp.join(root, 'default.txt'))

######################################################################

if __name__ == "__main__":
    main('/home/tm/nasrw/mk/MetaR-CNN/dataset/VOC2007')