import scipy.io as scio
import numpy as np
import cv2

from pathlib import Path


dir_path = Path('Y:/mk/dataset/결함/CrackForest-dataset/groundTruth/')
mat_file_paths = list(dir_path.glob('*.mat'))

mat_files = [scio.loadmat(path) for path in mat_file_paths]
segs = [mat['groundTruth']['Segmentation'][0,0] for mat in mat_files]
masks = [cv2.cvtColor(seg.astype('uint8'), cv2.COLOR_GRAY2BGRA) for seg in segs]
masks = [np.where(seg==1, 0, seg) for seg in masks]
masks = [np.where(seg==2, 100, seg ) for seg in masks]
masks = [np.where(seg==3, 170, seg ) for seg in masks]
masks = [np.where(seg==4, 250, seg ) for seg in masks]
for mask, path in zip(masks, mat_file_paths):
    save_path = Path('d:/crack/')/path.with_suffix('.png').name
    cv2.imwrite(str(save_path), mask)

print()