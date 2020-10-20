from functools import reduce
from numpy.lib.arraysetops import isin
from sklearn.metrics import jaccard_score
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import pandas as pd
import os.path as osp
import os
from nptyping import NDArray
from tqdm import tqdm
from typing import List, Union

'''
CVAT for image 2개의 iou를 비교하고 이미지와 csv파일 출력
'''

def load_annotation(xmlfile:str) -> ET.Element:
    in_file = open(xmlfile, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    return root

def polygon2mask(img_ET:ET.Element, labels:list) -> NDArray:
    height = int(img_ET.get('height'))
    width = int(img_ET.get('width'))

    img = Image.new('L', (width, height), 0)
    for child in list(img_ET):
        if child.tag == 'polygon':
            label = child.get('label')
            color = labels.index(label)
            polygon = list(map(float, child.get('points').replace(';',',').split(',')))
            ImageDraw.Draw(img).polygon(polygon, outline=color, fill=color)
    mask = np.array(img)
    return mask

def bbox2mask(img_ET:ET.Element, labels:list) -> NDArray:
    height = int(img_ET.get('height'))
    width = int(img_ET.get('width'))

    img = Image.new('L', (width, height), 0)
    for child in list(img_ET):
        if child.tag == 'box':
            label = child.get('label')
            color = labels.index(label)
            polygon = [child.get('xtl'), child.get('ytl'), 
                       child.get('xbr'), child.get('ytl'), 
                       child.get('xbr'), child.get('ybr'),
                       child.get('xtl'), child.get('ybr')]
            polygon = list(map(float, polygon))
            ImageDraw.Draw(img).polygon(polygon, outline=color, fill=color)
    mask = np.array(img)
    return mask

def cal_seg_iou(mask1, mask2, label_count) -> list:
    return jaccard_score(mask1, mask2, average=None, labels=list(range(1,1+label_count)))

def get_labels_color(root:Union[ET.Element, List[ET.Element]]) -> list:
    labels_color = []
    labels_color_ = lambda el: list(map(lambda x: ( x.find('name').text,  hex2rgb(x.find('color').text)), el.find('meta').find('task').find('labels')))
    if isinstance(root, ET.Element):
        labels_color.extend(labels_color_(root))
        labels_color = dict(labels_color)
    elif isinstance(root, list):
        labels_list = list(map(dict,map(labels_color_, root)))
        labels_list.append(dict(labels_color))
        labels_color = reduce(lambda x,y: dic_update(x,y), labels_list)
    return labels_color

def hex2rgb(v):
    v = v.lstrip('#')
    lv = len(v)
    return tuple(int(v[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def get_colors(root:ET.Element) -> list:
    colors = [(0,0,0)]
    colors.extend(list(map(lambda x: hex2rgb(x.find('color').text), root.find('meta').find('task').find('labels'))))
    return colors

def get_specific_img_name(full_path, range):
    path_split = full_path.split('/')[-range:]
    specific_img_name = '/'.join(path_split)
    return specific_img_name

def images_dict(root:ET.Element)->dict:
    el_dic = dict(map(lambda el: (get_specific_img_name(el.get('name'), 2).lower(), el), root.findall('image')))
    file_path_dic = dict(map(lambda el: (get_specific_img_name(el.get('name'), 2).lower(), el.get('name').lower()), root.findall('image')))
    return el_dic, file_path_dic

def cvt_id2rgb(id_array, colors):
    rgb_array = id_array
    for id, color in enumerate(colors):
        if id==0:
            continue
        assert not all(map(lambda x: x==color[0], color))
        rgb_array = np.where(rgb_array[...,:]==(id, id, id), color, rgb_array[...,:])
    return rgb_array

def expand_dims_pil(PIL_Image):
    return np.expand_dims(np.array(PIL_Image), axis=0)

def export_img_(gt_mask, wk_mask, gt_path, gt_path_dic, img_file_name, save_folder, annotation, colors, include_origin_img=False):
    rgb_gt_mask = cvt_id2rgb(expand_dims_pil(Image.fromarray(gt_mask).convert('RGB')),colors)
    rgb_gt_mask = np.where(rgb_gt_mask[...,:]==[0,0,0], np.nan,rgb_gt_mask)*0.7
    rgb_wk_mask = cvt_id2rgb(expand_dims_pil(Image.fromarray(wk_mask).convert('RGB')), colors)
    rgb_wk_mask = np.where(rgb_wk_mask[...,:]==[0,0,0], np.nan,rgb_wk_mask)*1.3
    img = np.concatenate([rgb_gt_mask, rgb_wk_mask], axis=0)
    if include_origin_img:
        origin_img = expand_dims_pil(Image.open(osp.join(osp.split(gt_path)[0], 'images', gt_path_dic[img_file_name])).convert('RGB'))
        img = np.concatenate([img, origin_img], axis=0)
    img = np.nanmean(img, axis=0)
    img = Image.fromarray(np.uint8(img), mode='RGB')
    name, _ = osp.splitext(img_file_name)
    new_name = f'{name}_{annotation}.jpg'
    save_path = osp.join(save_folder, 'images', new_name)
    os.makedirs(osp.split(save_path)[0], exist_ok=True)
    img.save(save_path)

def dic_update(dict1:dict, dict2:dict)-> dict:
    dict1.update(dict2)
    return dict1

def get_el_dict(path):
    if isinstance(path,list):
        root = list(map(load_annotation, path))
        el_dic, path_dic = list(reduce(lambda x,y: (dic_update(x[0],y[0]), dic_update(x[1],y[1])), map(images_dict,root)))
    elif isinstance(path, str):
        root = load_annotation(path)
        el_dic, path_dic = images_dict(root)
    else:
        raise TypeError
    return root, el_dic, path_dic

def main(gt_path, wk_path, save_folder, export_img=True, include_origin_img=False):
    '''
    export_img: 라벨 이미지를 출력할건지
    include_origin_img: export_img가 True라면 원본 이미지를 포함할건지.
                        이 인수가 Fasle일 경우 mask끼리만 연산한 이미지 출력
    '''
    gt_root, gt_el_dic, gt_path_dic = get_el_dict(gt_path)
    wk_root, wk_el_dic, _ = get_el_dict(wk_path)

    labels_color = get_labels_color(gt_root)
    labels = list(labels_color.keys())
    colors = list(labels_color.values())
    assert labels==list(get_labels_color(wk_root).keys()), '두 annotation 파일의 class가 다릅니다.'

    labels.insert(0, 'background')
    colors.insert(0, (0,0,0))

    if isinstance(gt_path, list):
        gt_path_ = gt_path[0]
    elif isinstance(gt_path, str):
        gt_path_ = gt_path
    else:
        raise TypeError

    seg_iou_df = pd.DataFrame(columns=labels, index=gt_el_dic.keys())
    bbox_iou_df = pd.DataFrame(columns=labels, index=gt_el_dic.keys())
    for img_file_name, gt_img_el in tqdm(gt_el_dic.items()):
        #get segmentation iou
        gt_mask = polygon2mask(gt_img_el, labels)
        wk_mask = polygon2mask(wk_el_dic[img_file_name], labels)
        seg_iou = cal_seg_iou(gt_mask.flatten(), wk_mask.flatten(), len(labels))
        seg_iou_df.loc[img_file_name] = list(seg_iou)
        if export_img:
            export_img_(gt_mask, wk_mask, gt_path_, gt_path_dic, img_file_name, save_folder, 'seg', colors, include_origin_img)

        #get bounding box iou
        gt_mask = bbox2mask(gt_img_el, labels)
        wk_mask = bbox2mask(wk_el_dic[img_file_name], labels)
        bbox_iou = cal_seg_iou(gt_mask.flatten(), wk_mask.flatten(), len(labels))
        bbox_iou_df.loc[img_file_name] = list(bbox_iou)
        if export_img:
            export_img_(gt_mask, wk_mask, gt_path_, gt_path_dic, img_file_name, save_folder, 'bbox', colors, include_origin_img)

    seg_iou_df = seg_iou_df.replace({0:np.nan})
    bbox_iou_df = bbox_iou_df.replace({0:np.nan})

    seg_iou_df = seg_iou_df.append(seg_iou_df.describe())
    seg_iou_des = pd.concat([seg_iou_df,seg_iou_df.T.describe().T], axis=1)
    bbox_iou_df = bbox_iou_df.append(bbox_iou_df.describe())
    bbox_iou_des = pd.concat([bbox_iou_df,bbox_iou_df.T.describe().T], axis=1)

    seg_iou_des.to_csv(osp.join(save_folder,'seg_iou.csv'), encoding='euc-kr')
    bbox_iou_des.to_csv(osp.join(save_folder,'bbox_iou.csv'), encoding='euc-kr')

if __name__ == "__main__":
    #정답 annotation
    gt_path1 = osp.expanduser('~/host/nasrw/mk/하수관로/test/gt_annotations_1.xml')
    gt_path2 = osp.expanduser('~/host/nasrw/mk/하수관로/test/gt_annotations_2.xml')

    #비교할 annotation
    wk_path1 = osp.expanduser('~/host/nasrw/mk/하수관로/test/wk_annotations_1.xml')
    wk_path2 = osp.expanduser('~/host/nasrw/mk/하수관로/test/wk_annotations_2.xml')

    save_folder = osp.expanduser('~/host/nasrw/mk/하수관로/test/report')

    main([gt_path1], [wk_path1], save_folder, export_img=True, include_origin_img=True)