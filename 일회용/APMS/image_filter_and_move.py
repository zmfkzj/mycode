from exif import Image
from pathlib import Path
from numpy.lib.type_check import real
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

def get_polar_field(image_size, real_size):
    img_h, img_w = image_size
    img_x = np.expand_dims(np.repeat(np.expand_dims(np.arange(img_w),axis=1),img_h,axis=1),axis=2)
    img_y = np.expand_dims(np.repeat(np.expand_dims(np.arange(img_h),axis=1),img_w,axis=1).T,axis=2)
    img_field = np.concatenate([img_y,img_x], axis=2)
    img_field = img_field/np .array([img_h,img_w])-0.5
    distance = np.sqrt(np.sum(np.square(img_field),axis=2))*get_distance(*real_size)
    angle = np.arctan(img_field[...,0]/img_field[...,1]) 
    angle = np.where(np.isnan(angle),0,angle)
    angle = np.where(img_field[...,1]<0,angle+np.pi,angle)
    polar_field = np.concatenate([np.expand_dims(distance,2),np.expand_dims(angle,2)],axis=2)
    return polar_field

def rotate(polar_field, north_angle):
    radian_angle = north_angle/180*np.pi
    polar_field[...,1] = polar_field[...,1]+radian_angle
    return polar_field

def cvt_polar2cartesian(polar_field):
    distance = polar_field[...,0]
    angle = polar_field[...,1]
    x = np.cos(angle)*distance
    y = np.sin(angle)*distance
    return np.concatenate([np.expand_dims(y,2),np.expand_dims(x,2)],axis=2)

# dir_path = Path()
# db_gps_info = pd.DataFrame(columns='section sample slab gps_lt gps_rb'.split())
# db_gps_info['gps_center'] = (db_gps_info['gps_lt'].astype(np.array)+ db_gps_info['gps_rb'].astype(np.array)).mean(axis=1)
# db_gps_info['gps_center'] = db_gps_info['gps_center'].map(lambda x,y: get_tm_coord(x,y))

# images = []
# for root,dirs,files in os.walk(dir_path):
#     for f in files:
#         if Path(f).suffix.lower() in ['.jpg','.png']:
#             images.append(Path(root)/f)

# for img in images:
#     img_tm = get_image_gps_info(img)
#     db_gps_info['new_image'] = db_gps_info['gps_center'].map(lambda db_tm: img if get_distance(db_tm,img_tm)<1 else np.nan)

# for slab in db_gps_info.itertuples:
#     if slab.new_image is not None:
#         new_path = Path(slab.section)/slab.sample/slab.new_image.name
#         copy(str(slab.new_image),str(new_path))
    
if __name__=="__main__":
    polar_field = get_polar_field((1080,1620),(60,160))
    rotated_polar_field = rotate(polar_field,79.83)
    cartesian_field = cvt_polar2cartesian(rotated_polar_field)
    lt = cartesian_field[0,0,:]
    rb = cartesian_field[-1,-1,:]
    pass