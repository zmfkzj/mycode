from exif import Image
from pathlib import Path
from pyproj import Transformer
from PIL import Image as PImage
from dtsummary.object import Bbox

import numpy as np
import matplotlib.pyplot as plt
import chardet
import json


def get_image_gps_info(image_path:Path):
    with open(image_path,'r+b') as f:
        img = Image(f)
    return get_tm_coord(img.gps_latitude,img.gps_longitude)

def get_tm_coord(latitude,longitude):
    gps_info = np.array((latitude,longitude))
    single_number_gps_info = np.sum(gps_info / np.array((1,60,3600)),axis=1)

    transformer = Transformer.from_crs('epsg:4737','epsg:5186')
    return transformer.transform(*single_number_gps_info)

def load2D(path2D):
    with open(path2D, 'r+b') as f:
        bytefile = f.read()
    encoding = chardet.detect(bytefile)['encoding']
    with open(path2D, 'r', encoding=encoding) as f:
        raw2D = json.load(f)
    return raw2D

def cal_GPSBbox(img_gps,bbox:Bbox, img_size):
    mmp = 13.7061/1000
    box_size =  np.array((bbox.h_a,bbox.w_a))
    gps_box_size = img_gps+box_size*mmp*(-1,1)

    lt_img_coord =  np.array((bbox.y1_a,bbox.x1_a))-np.array(img_size)/2
    lt_gps_coord = img_gps+lt_img_coord*mmp*(-1,1)

    return lt_gps_coord,gps_box_size

if __name__=='__main__':
    data = load2D('d:/ensemble/ensemble/costum_result_del.json')
    whole_objects = []
    img_gpses = []
    for d in data:
        filename = d['filename']
        img_size = (d['image_size']['height'],d['image_size']['width'])
        img_gps = get_image_gps_info(filename)
        for obj in d['objects']:
            if obj['label']=='Crater' and obj['confidence']>0.5:
                img_gpses.append(img_gps)
                center_coord, gps_bbox_size = cal_GPSBbox(img_gps,Bbox(img_size,**obj),img_size)
                whole_objects.append(center_coord)
    
    whole_x,whole_y = zip(*whole_objects)
    lats,lons = zip(*img_gpses)


    plt.figure(figsize=(10,10))
    plt.scatter(lons,lats,label='img_gps')
    plt.scatter(whole_y,whole_x, label='objects')
    plt.legend()
    plt.show()