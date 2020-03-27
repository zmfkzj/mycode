import cv2
import imageio as ii
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

imgtextfile = '/home/tm/Code/darknet/data/train.txt'
imgrootpath = '/home/tm/Code/darknet'
newpath = None


with open(imgtextfile, 'r') as f:
    imgpathlist = f.readlines()
imgpathlist = [i.rstrip('\n') for i in imgpathlist]
imgpathlist = [f'{imgrootpath}/{i}' for i in imgpathlist]

commonpath = os.path.commonpath(imgpathlist)
commondir = os.path.basename(commonpath)
newsavefolder = f'resize_{commondir}'
newsavepath = os.path.join(commonpath,'..',newsavefolder)
if not os.path.isdir(newsavepath):
    os.makedirs(newsavepath)

for imgpath in imgpathlist:
    img = ii.imread(imgpath)
    img = cv2.resize(img, dsize=(1000,1000))

    identitypath = os.path.relpath(imgpath, commonpath)
    newpath = os.path.join(newsavepath, identitypath)
    newpath = os.path.normpath(newpath)
    newpathdir = os.path.split(newpath)[0]
    if not os.path.isdir(newpathdir):
        os.makedirs(newpathdir)
    ii.imwrite(newpath, img)
    print(f'save - {newpath}')
print('image resize complete!')