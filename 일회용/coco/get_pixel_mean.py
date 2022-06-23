import numpy as np
import os.path

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

coco = COCO('j:/62Nas/mk/merged_dataset_coco/annotations/train.json')
image_dir = 'j:/62Nas/mk/merged_dataset_coco/images/'

images = coco.dataset['images']

pixel_total_sum = np.zeros(3)
pixel_total_size = 0
for img_info in tqdm(images):
    path = os.path.join(image_dir,img_info['file_name'])
    img = np.array(Image.open(path).convert('RGB'))
    pixel_total_sum += np.sum(np.sum(img,0),0)
    pixel_total_size += img.size//3

pixel_mean = pixel_total_sum / pixel_total_size
print(pixel_mean)
    
    