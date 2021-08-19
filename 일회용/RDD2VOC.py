from pathlib import Path
from collections import defaultdict
from shutil import copy
from tqdm import tqdm

import os

rdd_dataset_path = 'Z:\\62Nas\\NJY\\rdd dataset'

images = defaultdict(list)
annos = defaultdict(list)
for r,_,fs in os.walk(rdd_dataset_path):
    for f in fs:
        full_path = Path(r)/f
        if full_path.suffix == '.jpg':
            images[full_path.stem].append(full_path)
        elif full_path.suffix == '.xml':
            annos[full_path.stem].append(full_path)
        else:
            continue

inter_item = set(images) & set(annos)

anno_save_path = os.path.join(rdd_dataset_path,'merged_dataset/Annotations')
image_save_path = os.path.join(rdd_dataset_path,'merged_dataset/JPEGImages')
txt_save_path = os.path.join(rdd_dataset_path,'merged_dataset/ImageSets/Main')

os.makedirs(anno_save_path,exist_ok=True)
os.makedirs(image_save_path,exist_ok=True)
os.makedirs(txt_save_path,exist_ok=True)

# for id in tqdm(inter_item, desc='coping...'):
#     if not os.path.exists(os.path.join(image_save_path,str(Path(images[id][0]).name))):
#         copy(images[id][0],image_save_path)
#     if not os.path.exists(os.path.join(anno_save_path,str(Path(annos[id][0]).name))):
#         copy(annos[id][0],anno_save_path)

with open(os.path.join(txt_save_path,'default.txt'),'w') as f:
    f.write('\n'.join(inter_item))
