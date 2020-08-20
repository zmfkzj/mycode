import imageio
import imgaug as ia
import numpy as np
from memory_profiler import profile
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import random as rd
from itertools import repeat
import time


start = time.time()
# image = imageio.imread('~/nasrw/mk/MetaR-CNN/dataset/VOC2007/JPEGImages/132.jpg')
@profile
def load_img():
    image = np.ones((20000, 20000, 3), dtype=np.uint8)
    return image

@profile
def make_bbox(shape):

    h, w, _ = shape
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        BoundingBox(x1=rd.random()*w, x2=rd.random()*w, y1=rd.random()*h, y2=rd.random()*h),
        ], shape=image.shape)
    return bbs

image = load_img()
ia.seed(4)

aug_opt = [
iaa.Fliplr(0.5),
iaa.Flipud(0.5),
iaa.geometric.Rot90((0,3),keep_size=False),
iaa.Multiply((0.95,1.05), per_channel=True),
iaa.MultiplyHueAndSaturation((np.float16(0.5),np.float16(1.5)),per_channel=False),
iaa.MotionBlur(k=(3,5)),
iaa.Grayscale(),
iaa.RemoveCBAsByOutOfImageFraction(0.5),
iaa.ClipCBAsToImagePlanes()
]

@profile
def aug(img, aug_opt):
    image_aug = aug_opt(image=img)
    return image_aug

@profile
def aug_for(image, aug_opt):
    for idx, opt in enumerate(aug_opt):
        print(opt)
        print('start', time.time()-start)
        image_aug = aug(image, opt)
        print('end', time.time()-start, '\n')


aug_for(image, aug_opt)

@profile
def aug_seq(image, aug_opt):
    bbs = make_bbox(image.shape)
    images = list(repeat(image, 8))
    bbses = list(repeat(bbs, 8))
    seq = iaa.Sequential(aug_opt)
    # image_aug, bbs_aug = seq(images=images, bounding_boxes=bbses)
    image_aug = seq(images=images)
    del image_aug
    # del bbs_aug

# aug_seq(image, aug_opt)

@profile
def aug_for_images(image, aug_opt):
    images = list(repeat(image, 8))
    for idx, opt in enumerate(aug_opt):
        print(opt)
        image_aug = opt(images=images)

# aug_for_images(image, aug_opt)