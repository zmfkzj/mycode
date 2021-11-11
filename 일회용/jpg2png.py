from PIL import Image
from glob import glob
from pathlib import Path


dir_path = 'c:/Users/mkkim/Desktop/export/SQ_NEU_Dam/JPEGImages'
png_imgs = glob(f'{dir_path}/*.jp*')
for png_img in png_imgs:
    img = Image.open(png_img)
    path = Path(png_img)
    img.save(path.with_suffix('.png'))