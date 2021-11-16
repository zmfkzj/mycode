import pandas as pd
import numpy as np
from PIL import Image
import cv2

raw_data = pd.read_csv("z:/00.User/K/GIT_Namseoul University_Altitude25_20211009.txt",' ',header=None)
raw_data = raw_data.rename(columns = dict(zip(range(6),'x y z R G B'.split())))

x_min,x_max = raw_data['x'].min(), raw_data['x'].max()
y_min,y_max = raw_data['y'].min(), raw_data['y'].max()
xyRGB = raw_data['x y R G B'.split()]
step = 25

for x in np.arange(x_min,x_max,step):
    for y in np.arange(y_min,y_max,step):
        crop_data = raw_data.loc[(x<=raw_data['x'])&(raw_data['x']<(x+step))&(y<=raw_data['y'])&(raw_data['y']<(y+step)),:]
        if not crop_data.empty:
            bg = np.zeros((1024,1024,3))
            for i in ['x','y']:
                crop_data[i] = (crop_data[i]-crop_data[i].min())/(crop_data[i].max()-crop_data[i].min())*1023
                crop_data[i] = crop_data[i].map(np.around)
            crop_data = crop_data.groupby(['x','y']).mean().reset_index().astype(np.uint)
            bg[crop_data['y'],crop_data['x'],:] = crop_data['R G B'.split()].to_numpy()
            bg = bg.astype(np.uint8)
            bg = cv2.blur(bg,(5,5))
            crop_img = Image.fromarray(bg)
            crop_img.save(f'd:/새 폴더/{x}_{y}.jpg')

        