from functools import reduce
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Polygon,PolygonsOnImage
import os

import json
from pathlib import Path
from imgaug.parameters import Uniform
import numpy as np
import cv2
from pycocotools.coco import COCO
from copy import deepcopy
from shapely.geometry.polygon import Polygon as shapely_Polygon
from shapely.geometry.multipolygon import MultiPolygon as shapely_MultiPolygon
import matplotlib.pyplot as plt
import sys

limit_number = 100000
sys.setrecursionlimit(limit_number)

##############################################################################
coco_dataset = Path.home()/'pano16_coco'
# coco_dataset = Path('d:/pano16_coco')

crop_width, crop_height = 1000,600
count_per_img = 200
##############################################################################
image_dir = coco_dataset/'images'

coco = COCO(str(coco_dataset/'annotations/instances_default.json'))
coco_base_format = \
    {"licenses": [
        {"name": "",
        "id": 0,
        "url": "" } ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "" },
    "categories": [],
    "images": [],
    "annotations": [] }

coco_categories_base = \
    { "id": 1,
    "name": "Crack",
    "supercategory": "" }

coco_images_base = \
    { "id": 1,
    "width": 1000,
    "height": 600,
    "file_name": "A-1 (1)_균열.png",
    "license": 0,
    "flickr_url": "",
    "coco_url": "",
    "date_captured": 0 }

coco_annotations_base = \
    { "id": 3352,
    "image_id": 997,
    "category_id": 2,
    "segmentation": [ ],
    "area": 262578,
    "bbox":[],
    "iscrowd": 0,
    "attributes": { "occluded": False }
    }


ia.seed(1)
def cvtSeg(coco_seg):
    return [(coco_seg[idx],coco_seg[idx+1]) for idx in range(0,len(coco_seg),2)]

def is_divide_pt(x11,y11, x12,y12, x21,y21, x22,y22):

    '''input: 4 points
    output: True/False
    '''
    #  // line1 extension이 line2의 두 점을 양분하는지 검사..
    # 직선의 양분 판단
    f1= (x12-x11)*(y21-y11) - (y12-y11)*(x21-x11)
    f2= (x12-x11)*(y22-y11) - (y12-y11)*(x22-x11)
    if f1*f2 < 0 :
      return True
    else:
      return False

def is_cross(polygon1, polygon2):
    x11,y11 = polygon1[-1]
    x12,y12 = polygon2[0]
    x21,y21 = polygon2[-1]
    x22,y22 = polygon1[0]
    b1 = is_divide_pt(x11,y11, x12,y12, x21,y21, x22,y22)
    b2 = is_divide_pt(x21,y21, x22,y22, x11,y11, x12,y12)
    if b1 and b2:
        return True

def check_beetween_points(two_points,point):
    point1, point2 = two_points
    x_min, y_min= np.min(np.array([point1,point2]),axis=0)
    x_max, y_max= np.max(np.array([point1,point2]),axis=0)
    x,y = point
    if (x_min<=x) and (x<=x_max) and (y_min<=y) and (y<=y_max):
        x1,y1 = point1
        dx,dy = (np.array(point1)-np.array(point2)).tolist()
        if dx==0 and dy==0:
            if point1==point:
                return True
            else:
                return False
        elif dx==0:
            if point1[0]==point[0]:
                return True
            else:
                return False
        else:
            a = dy/dx
            b = y1-a*x1
            y_p = a*x+b
            if np.round(y,4)==np.round(y_p,4):
                return True
            else:
                return False
    else:
        return False

def make_dense_points(sparse_points, product=1):
    '''
    return (origin_marker, points)
    '''
    dense_points = []
    for idx in range(1,len(sparse_points)):
        d = np.sqrt(np.sum(np.square(np.array(sparse_points[idx-1])-np.array(sparse_points[idx]))))
        points = np.linspace(sparse_points[idx-1],sparse_points[idx],num=int(d)*product+1,endpoint=False).tolist()
        origin_marker = np.zeros(len(points))
        origin_marker[0] = 1
        dense_points.append(list(zip(origin_marker.astype(bool).tolist(),points)))
    merge_points = merge_list(dense_points)
    merge_points.append((True, sparse_points[-1]))
    return list(zip(*merge_points))

def get_nearest_points(polygon_points1, polygon_points2, product=1):
    markers1, dense_points1 = make_dense_points(polygon_points1, product)
    markers2, dense_points2 = make_dense_points(polygon_points2, product)
    dense_array1 = np.array(dense_points1)
    dense_array2 = np.array(dense_points2)

    dense_array1 = np.expand_dims(dense_array1,0).transpose(1,0,2)
    dense_array2 = np.expand_dims(dense_array2,0)
    distance = np.sqrt(np.sum(np.square(dense_array1-dense_array2),axis=2))
    idx1, idx2 = np.unravel_index(np.argmin(distance),distance.shape)

    return (dense_points1[int(idx1)], dense_points2[int(idx2)])

def get_nearest_index(polygon_points,point):
    return_idx = 0
    points = [polygon_points[-1]]+polygon_points
    for idx in range(1,len(points)):
        if check_beetween_points(points[idx-1:idx+1],point):
            return return_idx
        else:
            return_idx+=1
            continue
    plt.figure()
    plt.plot(*zip(*polygon_points))
    plt.scatter(*zip(*polygon_points))
    plt.scatter(*point)
    plt.show()
    raise Exception

def add_gap(polygon_points, gap=10):
    markers, dense_points = make_dense_points(polygon_points,product=10)
    new_points = dense_points[gap:-(gap+1)]
    new_markers = list(markers[gap:-(gap+1)])
    new_markers[0] = True
    new_markers[-1] = True
    return [point for marker,point in zip(new_markers,new_points) if marker]

def rm_duple_point(points):
    x,xs = points[0], points[1:]
    if not xs:
        return [x]
    elif x==xs[0]:
        return rm_duple_point(xs)
    else:
        return [x]+rm_duple_point(xs)

def get_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.sqrt(np.sum(np.square(point1-point2)))

def _merge_polygons(exterior, interiors):
    exterior = exterior+[exterior[0]]
    if not interiors:
        return exterior
    else:
        if len(interiors)==1:
            nearest_interior = interiors[0]+[interiors[0][0]]
            other_interiors = []
        else:
            nearest_points = [get_nearest_points(exterior,interior) for interior in interiors]
            distancese = [get_distance(*points) for points in nearest_points]
            sorted_interiors = sorted(zip(distancese, interiors),key=lambda d: d[0])
            (_,nearest_interior), other_interiors = sorted_interiors[0],[i for d,i in sorted_interiors[1:]]
            nearest_interior = nearest_interior+[nearest_interior[0]]

        exterior_nearest_point, interior_nearest_point = get_nearest_points(exterior,nearest_interior)

        exterior_start_index = get_nearest_index(exterior,exterior_nearest_point)
        interior_start_index = get_nearest_index(nearest_interior,interior_nearest_point)
        exterior_polygon = Polygon(exterior).change_first_point_by_index(exterior_start_index)
        interior_polygon = Polygon(nearest_interior).change_first_point_by_index(interior_start_index)

        exterior_points = [exterior_nearest_point]+exterior_polygon.coords.tolist()+[exterior_nearest_point]
        interior_points = [interior_nearest_point]+interior_polygon.coords.tolist()+[interior_nearest_point]

        exterior_points = rm_duple_point(exterior_points)
        exterior_points = add_gap(exterior_points, gap=1)
        interior_points = rm_duple_point(interior_points)
        interior_points = add_gap(interior_points, gap=1)

        new_exterior = exterior_points+interior_points
        while not Polygon(new_exterior).is_valid:
            interior_points = interior_points[::-1]
            new_exterior = exterior_points+interior_points
            if not Polygon(new_exterior).is_valid:
                exterior_points = add_gap(exterior_points, gap=1)
                interior_points = add_gap(interior_points, gap=1)

        new_exterior = exterior_points+interior_points

        if not Polygon(new_exterior).is_valid:
            plt.figure()
            plt.plot(*zip(*new_exterior))
            plt.scatter(*zip(*new_exterior), s=5)
            plt.scatter(*new_exterior[0])
            plt.show()
        return _merge_polygons(new_exterior,other_interiors)

def merge_interior_exterior_polygons(shapely_poly:shapely_Polygon, label):
    exterior = list(shapely_poly.exterior.coords)
    interiors = [list(inter.coords) for inter in shapely_poly.interiors]
    merged_exterior = _merge_polygons(exterior,interiors)

    return Polygon(merged_exterior,label)

def process_invalid_polygons(polygons:Polygon):
    shapely_poly = polygons.to_shapely_polygon().buffer(0)

    if isinstance(shapely_poly,shapely_Polygon):
        return [merge_interior_exterior_polygons(shapely_poly,polygons.label)]
    else:
        polys = []
        for collection in shapely_poly.geoms:
            if isinstance(collection,shapely_Polygon):
                polys.append(merge_interior_exterior_polygons(collection,polygons.label))
            elif isinstance(collection, shapely_MultiPolygon):
                for single_polygon in collection:
                    polys.append(merge_interior_exterior_polygons(single_polygon,polygons.label))
        return polys

def merge_list(list_in_list) :
    if len(list_in_list)>1:
        return reduce(lambda x,y:x+y, list_in_list)
    elif len(list_in_list)==1:
        return list_in_list[0]
    else:
        return []

def crop(image, polygons, position='uniform'):
    polygons = merge_list(polygons)
    polys = PolygonsOnImage(polygons,shape=image.shape)
    if position != 'uniform':
        t,l= position
        b,r = t+crop_height, l+crop_width
        top_crop = t
        right_crop = image.shape[1]-r if (image.shape[1]-r)>=0 else 0
        bottom_crop = image.shape[0]-b if (image.shape[0]-b)>=0 else 0
        left_crop = l
        px = (top_crop,right_crop,bottom_crop,left_crop)
        crop_augmenter = [ iaa.Crop(px,keep_size=False, sample_independently=False)]
    else:
        crop_augmenter = [iaa.CropToFixedSize(width=crop_width, height=crop_height,position='uniform')]


    seq = iaa.Sequential([
        *crop_augmenter,
        iaa.RemoveCBAsByOutOfImageFraction(.99)
    ])

    image_aug, polys_aug = seq(image=image, polygons=polys)
    return image_aug, polys_aug

coco_base = coco_base_format.copy()
def add_cat(cat_name):
    coco_cat = coco_categories_base.copy()
    coco_cat['id'] = len(coco_base['categories'])+1
    coco_cat['name'] = cat_name
    coco_base['categories'].append(coco_cat)

def add_image(image_name, width, height):
    coco_image = coco_images_base.copy()
    coco_image['id'] = len(coco_base['images'])+1
    coco_image['file_name'] = image_name
    coco_image['width'] = width
    coco_image['height'] = height
    coco_base['images'].append(coco_image)
    return coco_image['id']

def add_anno(polygon:Polygon, image_id):

    coco_anno = deepcopy(coco_annotations_base)
    coco_anno['id'] = len(coco_base['annotations'])+1
    coco_anno['segmentation'].append(polygon.coords.flatten().tolist())
    coco_anno['image_id'] = image_id
    coco_anno['category_id'] = polygon.label
    bbox = polygon.to_bounding_box()
    coco_anno['bbox'] = [float(c) for c in [bbox.x1, bbox.y1, bbox.width,bbox.height]]
    coco_anno['area'] = float( polygon.area )
    coco_base['annotations'].append(coco_anno)

coco_base['categories'] = list(coco.cats.values())

def divide_image(pano_image:np.ndarray,divide_size:tuple, stride_size:tuple):
    pano_h, pano_w = pano_image.shape[:2]
    for start_h in np.arange(pano_h,step=stride_size[0]):
        for start_w in np.arange(pano_w,step=stride_size[1]):
            yield (start_h,start_w)

for image_id in coco.getImgIds():
    coco_image = coco.loadImgs(image_id)[0]

    image_path = image_dir/coco_image['file_name']
    image = np.fromfile(image_path,np.uint8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)

    coco_annos = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    polygons = [Polygon(cvtSeg(anno['segmentation'][0]),label=anno['category_id']) for anno in coco_annos if anno['segmentation']]
    polygons = [process_invalid_polygons(poly) if not poly.is_valid else [poly] for poly in polygons]


    os.makedirs(image_dir/'../images_aug',exist_ok=True)
    isc=0
    totalc=0
    # while (isc<100) and (totalc<count_per_img):
    #     crop_image, crop_anno = crop(image, polygons)
    for start_h, start_w in divide_image(image,(crop_height, crop_width),(crop_height, crop_width)):
        crop_image, crop_anno = crop(image, polygons, position=(int(start_h), int(start_w)))
        result, encoded_image = cv2.imencode(image_path.suffix,crop_image)
        new_image_name = image_path.with_name(f'{image_path.stem}_{start_h}_{start_w}{image_path.suffix}').name
        # new_image_name = image_path.with_name(f'{image_path.stem}_{totalc}{image_path.suffix}').name

        if result:
            with open(image_dir/'../images_aug'/new_image_name, 'w+b') as f:
                encoded_image.tofile(f)
        
        image_id = add_image(new_image_name,crop_width, crop_height)
        if not crop_anno.empty:
            isc+=1
            for polygon in crop_anno.polygons:
                if polygon.coords.shape[0] != 0:
                    # add_anno(polygon,image_id)
                    for clip_polygon in polygon.clip_out_of_image(crop_image):
                        add_anno(clip_polygon,image_id)
        totalc+=1

os.makedirs(coco_dataset/'annotations_aug',exist_ok=True)
with open(coco_dataset/'annotations_aug/instances_default.json','w') as f:
    json.dump(coco_base,f)
        


