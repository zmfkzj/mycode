from os.path import *
import sys
sys.path.append(dirname(dirname(dirname(__file__))))

import pandas as pd
import numpy as np
import os.path as osp
import os
import cv2
from collections import defaultdict


root = osp.expanduser('~/nasrw/mk/dataset/도형/도형/')
path = osp.join(root,'gtpart_default.csv')

crop_root = osp.join(root,'..', 'crop')
os.makedirs(crop_root, exist_ok=True)

gt = pd.read_csv(path, encoding='euc-kr').rename(columns={'class':'classes'})

class_count = defaultdict(int)
for obj in gt.itertuples():
    img = cv2.imread(osp.join(root,obj.img))
    crop_img = img[int(obj.gt_top):int(obj.gt_bottom), int(obj.gt_left):int(obj.gt_right)]
    cv2.imwrite(osp.join(crop_root, '{}_{}.jpg'.format(obj.classes, class_count[obj.classes])), crop_img)
    class_count[obj.classes] += 1

