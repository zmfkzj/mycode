from exif import Image
from pathlib import Path
import pandas as pd
from pyproj import Transformer
import numpy as np
import os
from shutil import copy


def get_distance(tm1,tm2):
    return np.sqrt(np.sum(np.square(np.array(tm1)-np.array(tm2))))

def get_image_gps_info(image_path:Path):
    with open(image_path,'r+b') as f:
        img = Image(f)
    return get_tm_coord(img.gps_latitude,img.gps_longitude)

def get_tm_coord(latitude,longitude):
    gps_info = np.array((latitude,longitude))
    single_number_gps_info = np.sum(gps_info / np.array((1,60,3600)),axis=1)

    transformer = Transformer.from_crs('epsg:4737','epsg:5186')
    return transformer.transform(*single_number_gps_info)

def cal_FOV(self, sensor_size, focal_length, distance):
    '''
    sensor_size = width, height
    '''
    times = focal_length/distance
    w,h = np.array(sensor_size) / times / 1000
    return w,h

def get_field(w,h):
    np.arange(w*h*2).reshape([h,w,2])

def cvt_

dir_path = Path()
db_gps_info = pd.DataFrame(columns='section sample slab gps_lt gps_rb'.split())
db_gps_info['gps_center'] = (db_gps_info['gps_lt'].astype(np.array)+ db_gps_info['gps_rb'].astype(np.array)).mean(axis=1)
db_gps_info['gps_center'] = db_gps_info['gps_center'].map(lambda x,y: get_tm_coord(x,y))

images = []
for root,dirs,files in os.walk(dir_path):
    for f in files:
        if Path(f).suffix.lower() in ['.jpg','.png']:
            images.append(Path(root)/f)

for img in images:
    img_tm = get_image_gps_info(img)
    db_gps_info['new_image'] = db_gps_info['gps_center'].map(lambda db_tm: img if get_distance(db_tm,img_tm)<1 else np.nan)

for slab in db_gps_info.itertuples:
    if slab.new_image is not None:
        new_path = Path(slab.section)/slab.sample/slab.new_image.name
        copy(str(slab.new_image),str(new_path))
    
