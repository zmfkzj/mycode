import pandas as pd
import numpy as np
from PIL import Image as PImage
import cv2
from pathlib import Path
import piexif

raw_data = pd.read_csv("d:/DeepInspection_Namseoul University_Altitude50_Phantom4RTK_20210908.txt",' ',header=None)
raw_data = raw_data.rename(columns = dict(zip(range(6),'x y z R G B'.split())))

x_min,x_max = raw_data['x'].min(), raw_data['x'].max()
y_min,y_max = raw_data['y'].min(), raw_data['y'].max()
xyRGB = raw_data['x y R G B'.split()]
stride = 20
crop_size = 30

new_dir = Path('d:/3D_crop_blur_5')
if not new_dir.is_dir():
    new_dir.mkdir()

for x in np.arange(x_min,x_max,stride):
    for y in np.arange(y_min,y_max,stride):
        crop_data = raw_data.loc[(x<=raw_data['x'])&(raw_data['x']<(x+crop_size))&(y<=raw_data['y'])&(raw_data['y']<(y+crop_size)),:]
        if not crop_data.empty:
            bg = np.zeros((1024,1024,3))
            for i in ['x','y']:
                crop_data[i] = (crop_data[i]-crop_data[i].min())/(crop_data[i].max()-crop_data[i].min())*1023
                crop_data[i] = crop_data[i].map(np.around)
            crop_data = crop_data.groupby(['x','y']).mean().reset_index().astype(np.uint)
            bg[crop_data['y'],crop_data['x'],:] = crop_data['R G B'.split()].to_numpy()
            bg = bg.astype(np.uint8)
            bg = cv2.blur(bg,(5,5))
            crop_img = PImage.fromarray(bg)
            exif_dict = {'GPS':{}}
            exif_dict['GPS'][piexif.GPSIFD.GPSAreaInformation] = f'{x},{y},{x+crop_size},{y+crop_size}'.encode('ascii')
            
            exif_bytes = piexif.dump(exif_dict)
            crop_img.save(new_dir/f'{x}_{y}.jpg',exif=exif_bytes)


        