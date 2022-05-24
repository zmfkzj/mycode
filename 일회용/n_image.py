'''
각 폴더에서 n개 이미지만 남기고 삭제
'''

from pathlib import Path
import random
import os

dir = Path('d:/XAI/fod')
n = 20

dirs = dir.glob('*')
for d in dirs:
    if d.is_dir():
        imgs = list(d.glob('*.jpg'))
        random.shuffle(imgs)
        rm_imgs = imgs[n:]

        for img in rm_imgs:
            os.remove(str(img))