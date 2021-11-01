from exif import Image
from pathlib import Path
import pandas as pd
from pyproj import Transformer
import numpy as np
import os
from shutil import copy
import psycopg2 as ps
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="d:/upload_images/")
    parser.add_argument("--output_dir", type=str, default="d:/moved_images")
    parser.add_argument("--sensor_size", type=tuple, default=(8.8,13.2),help='h,w')
    parser.add_argument("--focal_length", type=float, default=24)
    parser.add_argument("--distance", type=float, default=50000)
    parser.add_argument("--north_angle", type=float, default=78.17,help='이미지 12시 기준으로 실제 북쪽이 몇도에 있는지')
    return parser.parse_args()

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

def cal_FOV(sensor_size, focal_length, distance):
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

def isin_image(img_ltrb, db_ltrb):
    img_min = np.min(np.array(img_ltrb),axis=0)
    img_max = np.max(np.array(img_ltrb),axis=0)
    db_lt, db_rb = db_ltrb
    db_lt = get_tm_coord(*db_lt)
    db_rb = get_tm_coord(*db_rb)
    db_min = np.min(np.array((db_lt, db_rb)),axis=0)
    db_max = np.max(np.array((db_lt, db_rb)),axis=0)

    if np.all(img_min<=db_min) and np.all(img_max>=db_max):
        return True
    else:
        return False

arg = parser()

db = []
with ps.connect(host='localhost', dbname='aroad', user='postgres', password='namchanho1!@',port=5432) as conn:  # db에 접속
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM gps_test")
        for data in cur:
            db.append(pd.Series(data))

db_gps_info = pd.DataFrame(db).rename(columns={0:'lt_lat',1:'lt_lon',2:'rb_lat',3:'rb_lon',4:'slab_id'})

images = []
for root,dirs,files in os.walk(arg.input_dir):
    for f in files:
        if Path(f).suffix.lower() in ['.jpg','.png']:
            images.append(Path(root)/f)

for img in images:
    img_tm = get_image_gps_info(img)
    real_size = cal_FOV(arg.sensor_size, arg.focal_length,arg.distance)

    with open(img,'r+b') as f:
        img_meta = Image(f)
    
    polar_field = get_polar_field((img_meta.pixel_y_dimension,img_meta.pixel_x_dimension),real_size)
    rotated_polar_field = rotate(polar_field,arg.north_angle)
    cartesian_field = cvt_polar2cartesian(rotated_polar_field)
    lt = cartesian_field[0,0,:]*real_size+img_tm
    rb = cartesian_field[-1,-1,:]*real_size+img_tm

    for slab in db_gps_info.itertuples():
        if isin_image((lt,rb),((slab.lt_lat,slab.lt_lon),(slab.rb_lat,slab.rb_lon))):
            new_path = Path(arg.output_dir)/f'{slab.slab_id}{Path(img).suffix}'
            os.makedirs(str(new_path.parent),exist_ok=True)
            copy(str(img),str(new_path))