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

def load_images(dir_path:Path):
    if isinstance(dir_path,str):
        dir_path = Path(dir_path)
    imgs_path = []
    for ext in ['jpg','png']:
        imgs_path.extend(dir_path.glob(f"*.{ext}"))
    img_and_gps = [(np.array(PImage.open(path)),get_image_gps_info(path)) for path in imgs_path]
    return img_and_gps

def load2D(path2D):
    with open(path2D, 'r+b') as f:
        bytefile = f.read()
    encoding = chardet.detect(bytefile)['encoding']
    with open(path2D, 'r', encoding=encoding) as f:
        raw2D = json.load(f)
    return raw2D

def process2D(raw2D, img_tm_coord):
    whole_objects = []
    for img in raw2D:
        objects = img['objects']
        img_size = (img['image_size']['height'],img['image_size']['width'])
        for obj in objects:
            bbox = Bbox(img_size,**obj)
            obj['voc_bbox'] = cal_GPSBbox(bbox,img_size)
            del obj['yolo_bbox']
            whole_objects.append(obj)
    return whole_objects

def cal_GPSBbox(bbox:Bbox, img_size):
    return np.array((bbox.xc_a,bbox.yc_a))-img_size[::-1]/2

def arrange_image(img_and_gps):
    imgs, gpses = zip(*img_and_gps)
    lats,lons = zip(*gpses)
    data = load2D('d:/ensemble/ensemble/costum_result_del.json')
    whole_objs = []
    for gps in gpses:
        whole_objs.extend(process2D(data,gps))

    plt.scatter(lats,lons)
    plt.show()




if __name__=='__main__':
    img_and_gps = load_images('d:/ensemble/ensemble/images_crater/')
    arrange_image(img_and_gps)