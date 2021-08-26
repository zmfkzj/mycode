import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Polygon,PolygonsOnImage
import os

import json
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO
from copy import deepcopy
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon as shapely_Polygon
from shapely.validation import make_valid,explain_validity
from shapely.ops import nearest_points as get_nearest_points

##############################################################################
coco_dataset = Path('d:/pano16_coco')
image_dir = coco_dataset/'images'

coco = COCO(str(coco_dataset/'annotations/instances_default.json'))
crop_width, crop_height = 1000,600

##############################################################################
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

def signed_area(pr2):
     """Return the signed area enclosed by a ring using the linear time
     algorithm at http://www.cgafaq.info/wiki/Polygon_Area. A value >= 0
     indicates a counter-clockwise oriented ring."""
     xs, ys = map(list, zip(*pr2))
     xs.append(xs[1])
     ys.append(ys[1])
     return sum(xs[i]*(ys[i+1]-ys[i-1]) for i in range(1, len(pr2)))/2.0

def rotation_dir(pr):
     signedarea = signed_area(pr)
     if signedarea > 0:
         return "anti-clockwise"
     elif signedarea < 0:
         return "clockwise"
     else:
         return "UNKNOWN"

def merge_interior_exterior_polygons(shapely_poly:shapely_Polygon, label):
    exterior = list(shapely_poly.exterior.coords)
    for inter in shapely_poly.interiors:
        interior = list(inter.coords)
        exterior_nearest_point, interior_nearest_point = [list(point.coords) for point in get_nearest_points(shapely_Polygon(exterior),shapely_Polygon(interior))]
        exterior_polygon = Polygon(exterior,label).change_first_point_by_coords(*exterior_nearest_point,max_distance=None)
        interior_polygon = Polygon(interior,label).change_first_point_by_coords(*interior_nearest_point,max_distance=None)

        exterior_points = [exterior_nearest_point]+exterior_polygon.coords.tolist()+[exterior_nearest_point]
        interior_points = [interior_nearest_point]+interior_polygon.coords.tolist()+[interior_nearest_point]
        
        exterior_dir = rotation_dir(exterior_points)
        interior_dir = rotation_dir(interior_points)
        if exterior_dir==interior_dir:
            interior_points = interior_points[::-1]

        new_points = exterior_points+interior_points
        new_polygon = Polygon(new_points)
        exterior = new_polygon.coords.tolist()
    return Polygon(exterior,label)

def process_invalid_polygons(polygons:Polygon):
    shapely_poly = polygons.to_shapely_polygon().buffer(0)

    if isinstance(shapely_poly,shapely_Polygon):
        return merge_interior_exterior_polygons(shapely_poly,polygons.label)
    else:
        polys = []
        for single_polygon in list(shapely_poly):
            polys.append(merge_interior_exterior_polygons(single_polygon,polygons.label))
        return polys

def merge_list(list_in_list) :
    if list_in_list:
        x,xs = list_in_list[0], list_in_list[1:]
        if xs:
            return x+merge_list(xs)
        else:
            return x
    else:
        return []

def crop(image, polygons):
    polygons = [process_invalid_polygons(poly) if not poly.is_valid else [poly] for poly in polygons]
    polygons = merge_list(polygons)
    polys = PolygonsOnImage(polygons,shape=image.shape)

    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=crop_width, height=crop_height),
        iaa.RemoveCBAsByOutOfImageFraction(.9)
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

for image_id in coco.getImgIds():
    coco_image = coco.loadImgs(image_id)[0]

    image_path = image_dir/coco_image['file_name']
    image = np.fromfile(image_path,np.uint8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)

    coco_annos = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    polygons = [Polygon(cvtSeg(anno['segmentation'][0]),label=anno['category_id']) for anno in coco_annos if anno['segmentation']]
    os.makedirs(image_dir/'../images_aug',exist_ok=True)
    isc=0
    totalc=0
    while (isc<100) and (totalc<200):
        crop_image, crop_anno = crop(image, polygons)
        result, encoded_image = cv2.imencode(image_path.suffix,crop_image)
        new_image_name = image_path.with_name(f'{image_path.stem}_{totalc}{image_path.suffix}').name

        if result:
            with open(image_dir/'../images_aug'/new_image_name, 'w+b') as f:
                encoded_image.tofile(f)
        
        image_id = add_image(new_image_name,crop_width, crop_height)
        if not crop_anno.empty:
            isc+=1
            for polygon in crop_anno.polygons:
                if polygon.coords.shape[0] != 0:
                    polygon = make_valid(polygon)
                    add_anno(polygon,image_id)
                    # for clip_polygon in polygon.clip_out_of_image(crop_image):
                    #     add_anno(clip_polygon,image_id)
        totalc+=1

os.makedirs(coco_dataset/'annotations_aug',exist_ok=True)
with open(coco_dataset/'annotations_aug/instance_default.json','w') as f:
    json.dump(coco_base,f)
        


