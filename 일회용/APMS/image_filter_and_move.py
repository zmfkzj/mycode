from exif import Image
# import foilum as fl
import pathlib as fl
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
    parser.add_argument("--distance", type=str, default=3)
    parser.add_argument("--test", type=float, default=False,help='test 중?')
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


def cal_gps_num(degree):
    return np.sum(np.array(degree) / np.array([1,60,3600]))


arg = parser()

db = []
with ps.connect(host='localhost', dbname='aroad', user='postgres', password='namchanho1!@',port=5432) as conn:  # db에 접속
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM gps_test")
        for data in cur:
            db.append(pd.Series(data))

db_gps_info = pd.DataFrame(db).rename(columns={0:'ct_lat',1:'ct_lon',2:'slab_id'})
db_gps_info = db_gps_info.apply(lambda x: pd.Series(get_tm_coord(*x[:2])+(x[2],),index=['ct_lat','ct_lon','slab_id']) ,axis=1)


images = []
for root,dirs,files in os.walk(arg.input_dir):
    for f in files:
        if Path(f).suffix.lower() in ['.jpg','.png']:
            images.append(Path(root)/f)

if arg.test:
    map = fl.Map([36.90833,127.1408],zoom_start=15,max_zoom=22)

for img in images:
    img_tm = get_image_gps_info(img)

    if arg.test:
        transformer = Transformer.from_crs('epsg:5186','epsg:4737')
        img_gps = transformer.transform(*img_tm)

        fl.Marker(img_gps,icon=fl.Icon(color='red')).add_to(map)

    for slab in db_gps_info.itertuples():
        if get_distance(img_tm,(slab.ct_lat, slab.ct_lon))<arg.distance:
            new_path = Path(arg.output_dir)/f'{int(slab.slab_id)}{Path(img).suffix.lower()}'
            if arg.test:
                transformer = Transformer.from_crs('epsg:5186','epsg:4737')
                img_gps = transformer.transform(*img_tm)

                fl.Marker(img_gps,icon=fl.Icon(color='green')).add_to(map)
            else:
                os.makedirs(str(new_path.parent),exist_ok=True)
                copy(str(img),str(new_path))

if arg.test:
    for slab in db_gps_info.itertuples():
        transformer = Transformer.from_crs('epsg:5186','epsg:4737')
        db_gps = transformer.transform(slab.ct_lat, slab.ct_lon)

        fl.Marker(db_gps).add_to(map)

if arg.test:
    map.save('c:/map.html')