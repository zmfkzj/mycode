from exif import Image as eImage
from pathlib import Path
import pandas as pd
from pyproj import Transformer
import numpy as np
import os
from shutil import copy
import psycopg2 as ps
import argparse
import cv2
from PIL import Image as PImage


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="j:/00.User/Yun/input/")
    parser.add_argument("--output_dir", type=str, default="D:/APMS_output")
    parser.add_argument("--distance", type=str, default=2.5)
    return parser.parse_args()


def get_distance(tm1,tm2):
    diff = np.array(tm1)-np.array(tm2)
    return np.sqrt(np.sum(np.square(diff))), *diff


def get_image_gps_info(image_path, file_list):
    image_path = str(image_path)
    img_filename = image_path.split('_')[-5]

    for images_gps in file_list:
        for img_info in images_gps.itertuples():
            if img_filename == str(img_info.파일명):
                return get_tm_coord(img_info.위도, img_info.경도)
    return False


def get_tm_coord(latitude,longitude):
    gps_info = np.array((latitude,longitude))
    # single_number_gps_info = np.sum(gps_info / np.array((1,60,3600)),axis=1)

    transformer = Transformer.from_crs('epsg:4737','epsg:5186')
    return transformer.transform(*gps_info)


def cal_gps_num(degree):
    return np.sum(np.array(degree) / np.array([1,60,3600]))


arg = parser()
db_gps_info = pd.read_csv('j:/00.User/Yun/DB_gps_info_211122_2.csv', encoding='cp949')
images_gps_df = pd.read_csv('j:/00.User/Yun/Crack_GPS\\균열2차.csv', encoding='cp949')
images_gps_df2 = pd.read_csv('j:/00.User/Yun/Crack_GPS\\균열3차.csv', encoding='cp949')
images_gps_df3 = pd.read_csv('j:/00.User/Yun/Crack_GPS\\균열4차.csv', encoding='cp949')
images_gps_df4 = pd.read_csv('j:/00.User/Yun/Crack_GPS\\균열5차.csv', encoding='cp949')
# file_list = [images_gps_df, images_gps_df2, images_gps_df3, images_gps_df4]
file_list = [images_gps_df]
for images_gps_df in file_list:
    for col in images_gps_df.columns:
        images_gps_df.rename(columns={col : col.strip()}, inplace=True)

crop_img_size_w = 2766
crop_img_size_h = 2766
right_eye = '19318210'
left_eye = '19318211'
images = []
full_pixel = True
# gps 계산하느라 만듬
############################

# df = pd.read_csv('./Crack_GPS/균열5차.csv', encoding='cp949')
# df2 = pd.read_csv('./tmp.csv', encoding='cp949')
# for col in df.columns:
#     df.rename(columns={col : col.strip()}, inplace=True)
#
# for data in df.itertuples():
#     for data2 in df2.itertuples():
#         if str(data.파일명)[-4:] == str(data2.이미지넘버2):
#             lat = data.위도
#             long = data.경도
#             print(f'({lat}, {long})')
#############################

for root,dirs,files in os.walk(arg.input_dir):
    for f in files:
        if Path(f).suffix.lower() in ['.jpg','.png'] and f.split('_')[-2] == right_eye:
            images.append(Path(root)/f)

#  gps 계산하느라 만듬
############################
# 미터계 계산
# for data in df2.itertuples():
#     latlong = data.위도
#     latlong = str(latlong)
#     lat = latlong.split(',')[0][1:]
#     long = latlong.split(',')[1].strip()[:-1]
#     print(get_tm_coord(lat, long))
#
# 거리계산
# for data in df2.itertuples():
#     d1 = str(data.미터계)
#     d1_lat = d1.split(',')[0][1:]
#     d1_lat = float(d1_lat)
#     d1_long = d1.split(',')[1].strip()[:-1]
#     d1_long = float(d1_long)
#     d2 = str(data.미터계2)
#     d2_lat = d2.split(',')[0][1:]
#     d2_lat = float(d2_lat)
#     d2_long = d2.split(',')[1].strip()[:-1]
#     d2_long = float(d2_long)
#
#     d, _, _ = get_distance((d1_lat, d1_long), (d2_lat, d2_long))
#     print(d)

#############################
def is_in_ellipse(diff, distance, img_size):
    # 타원 영역 안에 있는지 확인
    rate = img_size[1]/img_size[0]
    vertical = distance
    horizontal = vertical*rate
    y,x = diff

    return x**2/horizontal**2 + y**2/vertical**2 <= 1

for img_path in images:
    img_tm = get_image_gps_info(img_path, file_list)
    img_num = str(img_path).split('_')[-6]
    img = PImage.open(str(img_path))
    img = np.array(img)
    if img_tm == False:
        continue

    for slab in db_gps_info.itertuples():
        d,d_lat,d_lon = get_distance(img_tm,(slab.ct_lat, slab.ct_lon))  # 미터

        dw = d_lat*1000/2.01  # 이미지 gps와 slab gps의 거리차이(pixel)
        dh = d_lon*1000/2.01
        # if d<arg.distance:
        if is_in_ellipse((d_lon,d_lat),arg.distance,img.shape):
            new_path = Path(arg.output_dir)/f'{int(img_num)}-{int(slab.slab_id)}distance{round(d, 3)}{Path(img_path).suffix.lower()}'
            nonduple_new_path = Path(arg.output_dir) / f'{int(slab.slab_id)}_{Path(img_path).suffix.lower()}'
            os.makedirs(str(new_path.parent),exist_ok=True)
            h,w = img.shape[:2]

            img_center_w = w//2
            img_center_h = h//2
            slab_w_in_img = round(img_center_w - dw)
            slab_h_in_img = round(img_center_h - dh)

            # 무조건 2766x2766으로 자름
            if full_pixel:
                if slab_w_in_img - (crop_img_size_w//2) < 0:  # slab의 gps가 이미지의 너무 왼쪽에 있을 때
                    crop_w_start = 0
                    crop_w_end = crop_w_start + crop_img_size_w
                elif slab_w_in_img + (crop_img_size_w//2) > w:  # slab의 gps가 이미지의 너무 오른쪽에 있을 때
                    crop_w_end = w
                    crop_w_start = crop_w_end - crop_img_size_w
                else:
                    crop_w_start = slab_w_in_img - (crop_img_size_w//2)
                    crop_w_end = slab_w_in_img + (crop_img_size_w//2)

                if slab_h_in_img - (crop_img_size_h//2) < 0:  # slab의 gps가 이미지의 너무 위쪽에 있을 때
                    crop_h_start = 0
                    crop_h_end = crop_h_start + crop_img_size_h
                elif slab_h_in_img + (crop_img_size_h//2) > h:  # slab의 gps가 이미지의 너무 아래쪽에 있을 때
                    crop_h_end = h
                    crop_h_start = crop_h_end - crop_img_size_h
                else:
                    crop_h_start = slab_h_in_img - (crop_img_size_h//2)
                    crop_h_end = slab_h_in_img + (crop_img_size_h//2)

                crop_img = img[crop_h_start:crop_h_end, crop_w_start:crop_w_end, ::-1]
                cv2.imwrite(str(new_path), crop_img)
                cv2.imwrite(str(nonduple_new_path), crop_img)

            # pixel값 많이 바뀜(image gps와 slab gps의 차이가 클 때)
            else:
                if slab_w_in_img - (crop_img_size_w//2) < 0:  # slab의 gps가 이미지의 너무 왼쪽에 있을 때
                    crop_w_start = 0
                    crop_w_end = slab_w_in_img + (crop_img_size_w//2)
                elif slab_w_in_img + (crop_img_size_w//2) > w:  # slab의 gps가 이미지의 너무 오른쪽에 있을 때
                    crop_w_start = slab_w_in_img - (crop_img_size_w//2)
                    crop_w_end = w
                else:
                    crop_w_start = slab_w_in_img - (crop_img_size_w//2)
                    crop_w_end = slab_w_in_img + (crop_img_size_w//2)

                if slab_h_in_img - (crop_img_size_h//2) < 0:  # slab의 gps가 이미지의 너무 위쪽에 있을 때
                    crop_h_start = 0
                    crop_h_end = slab_h_in_img + (crop_img_size_h//2)
                elif slab_h_in_img + (crop_img_size_h//2) > h:  # slab의 gps가 이미지의 너무 아래쪽에 있을 때
                    crop_h_start = slab_h_in_img - (crop_img_size_h//2)
                    crop_h_end = h
                else:
                    crop_h_start = slab_h_in_img - (crop_img_size_h//2)
                    crop_h_end = slab_h_in_img + (crop_img_size_h//2)
                crop_img = img[crop_h_start:crop_h_end, crop_w_start:crop_w_end, ::-1]

                # if len(crop_img) == crop_img_size_h and len(crop_img[0]) == crop_img_size_w:
                cv2.imwrite(str(new_path), crop_img)
                cv2.imwrite(str(nonduple_new_path), crop_img)