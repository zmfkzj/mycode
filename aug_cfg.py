from imgaug import augmenters as iaa
import numpy as np
import copy

share_aug_option = [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.geometric.Rot90((0,3),keep_size=False),
        iaa.Multiply((0.9,1.1), per_channel=True),
        iaa.MultiplyHueAndSaturation((0.5,1.5),per_channel=False),
        iaa.MotionBlur(k=(1,5)),
        iaa.Grayscale((0.0,1.0))
        ]

def seq_Pad(downscale, shape):
    add_aug_option = [
        iaa.Pad(percent=(0,0.5), keep_size=False),
        iaa.Resize({'height': downscale[0], 'width': downscale[1]})
        ]
    merge_aug_option = copy.deepcopy(add_aug_option)
    merge_aug_option.extend(share_aug_option)
    seq = iaa.Sequential(merge_aug_option)
    return seq

def seq_Crop(downscale, shape):
    add_aug_option = [
        iaa.Crop(percent=(0,0.2),keep_size=False),
        iaa.Resize({'height': downscale[0], 'width': downscale[1]})
        ]
    merge_aug_option = copy.deepcopy(add_aug_option)
    merge_aug_option.extend(share_aug_option)
    seq = iaa.Sequential(merge_aug_option)
    return seq

def seq_CropToFixedsize_Crop(downscale, shape):
    crop_size = int(np.array(shape[:2]).min() * 1)
    add_aug_option = [
        iaa.CropToFixedSize(width=crop_size, height=crop_size),
        iaa.Crop(percent=(0,0.5),keep_size=False),
        iaa.Resize({'height': downscale[0], 'width': downscale[1]})
        ]
    merge_aug_option = copy.deepcopy(add_aug_option)
    merge_aug_option.extend(share_aug_option)
    seq = iaa.Sequential(merge_aug_option)
    return seq

def seq_CropToFixedsize_Pad(downscale, shape):
    crop_size = int(np.array(shape[:2]).min() * 1)
    add_aug_option = [
        iaa.CropToFixedSize(width=crop_size, height=crop_size),
        iaa.Pad(percent=(0,0.5), keep_size=False),
        iaa.Resize({'height': downscale[0], 'width': downscale[1]})
        ]
    merge_aug_option = copy.deepcopy(add_aug_option)
    merge_aug_option.extend(share_aug_option)
    seq = iaa.Sequential(merge_aug_option)
    return seq
bbox_color = {'0':(255,0,0),
        '1':(0,255,0),
        '2':(0,0,255),
        '3':(255,255,0),
        '4':(255,0,255),
        '5':(0,255,255),
        '6':(255,255,255),
        '7':(25,200,180)}

def add_resize(seq, target_size, shape):
    minsize = min(shape[:2])
    new_size_ratio = shape[:2]/minsize
    new_size = [int(i) for i in min(target_size)*new_size_ratio]
    seq.insert(0,iaa.Resize({'height': new_size[0], 'width': new_size[1]}))
    return seq

def cal_newsize(downscale, shape):
    minsize = min(shape[:2])
    new_size_ratio = np.array(shape[:2])/minsize
    new_size = [int(i) for i in min(downscale)*new_size_ratio]
    return new_size