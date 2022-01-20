from os import PathLike
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import json
import chardet
from copy import deepcopy
from pathlib import Path
import cv2
from MLobject import Bbox
from pyproj import Transformer
# from exif import Image
import imgaug as ia
import imgaug.augmenters as iaa

class EnsembleDataManager:
    def __init__(self, path2D:PathLike, path3D:PathLike, cats:Dict[int,str], camera_spec:dict,north_angle:float,gps_info_path) -> None:
        self.path2D = path2D
        self.path3D = path3D
        self.cats = cats
        self.real_size = self.cal_FOV(**camera_spec)
        self.north_angle = north_angle
        self.images_gps_df = pd.read_csv(gps_info_path, encoding='cp949')
        self.load2D()
        self.load3D()
    
    @staticmethod
    def get_distance(tm1,tm2):
        return np.sqrt(np.sum(np.square(np.array(tm1)-np.array(tm2))))

    # def get_image_gps_info(self, image_path:Path):
    #     with open(image_path,'r+b') as f:
    #         img = Image(f)
    #     return self.get_tm_coord(img.gps_latitude,img.gps_longitude)

    def get_image_gps_info(self, image_path):
        image_path = str(image_path)

        for img_info in self.images_gps_df.itertuples():
            if str(img_info.파일명).strip() in image_path:
                return self.get_tm_coord(img_info.위도, img_info.경도)
        return False

    def get_tm_coord(self, latitude,longitude):
        gps_info = np.array((latitude,longitude))
        # gps_info = np.sum(gps_info / np.array((1,60,3600)),axis=1)

        transformer = Transformer.from_crs('epsg:4737','epsg:5186')
        return transformer.transform(*gps_info)

    def cal_FOV(self, sensor_size, focal_length, distance):
        '''
        sensor_size = height, width
        '''
        times = focal_length/distance
        return np.array(sensor_size) / times / 1000
    
    def cal_GPSBbox(self, img_tm_coord:Tuple[float,float], bbox:Bbox, img_size, fov):
        mpp = np.mean(fov/img_size)
        coord_from_center = np.array(bbox.voc[::-1])-np.tile(img_size,2)/2
        meter_from_center = coord_from_center * mpp*[-1,1,-1,1]
        return meter_from_center + np.tile(img_tm_coord,2)

    def load2D(self):
        with open(self.path2D, 'r+b') as f:
            bytefile = f.read()
        encoding = chardet.detect(bytefile)['encoding']
        with open(self.path2D, 'r', encoding=encoding) as f:
            self.raw2D = json.load(f)
        self.process2D()
    
    def process2D(self):
        data = deepcopy(self.raw2D)
        whole_objects = []
        for img in data:
            filename = img['filename']
            objects = img['objects']
            self.img_tm_coord = self.get_image_gps_info(Path(self.path2D).parent/'images'/Path(filename).name)
            if not self.img_tm_coord:
                continue
            else:
                img_size = (img['image_size']['height'],img['image_size']['width'])
                bboxes = []
                for obj in objects:
                    bbox = Bbox(img_size,**obj)
                    bbox = ia.BoundingBox(*bbox.voc,label=bbox.label)
                    bboxes.append(bbox)
                bboxesOnImg = ia.BoundingBoxesOnImage(bboxes,img_size)
                seq = iaa.Rotate(rotate=self.north_angle)
                bbs_aug = seq(bounding_boxes=bboxesOnImg)
                for obj,bbox in zip(objects,bbs_aug):
                    bbox = Bbox(img_size,voc_bbox=(bbox.x1-50,bbox.y1-50,bbox.x2+50,bbox.y2+50),label=bbox.label)
                    obj['y2'], obj['x2'], obj['y1'], obj['x1'] = self.cal_GPSBbox(self.img_tm_coord,bbox,img_size,self.real_size)
                    whole_objects.append(obj)

        self.data2D = pd.DataFrame([pd.Series(obj) for obj in whole_objects]).rename(columns={'confidence':'conf','label':'cat'})
        self.data2D = self.data2D.reindex(columns='x1 y1 x2 y2 conf cat'.split())

        # center_data = []
        # for x in self.data2D.itertuples():
        #     if x.conf>0.5:
        #         center_data.append((x.x1+(x.x2-x.x1)/2,x.y1+(x.y2-x.y1)/2, x.cat))
        # x_c, y_c, cat = zip(*center_data)
        # data = {'x_center': x_c, 'y_center':y_c, 'cat':cat}
        # df = pd.DataFrame(data)
        # df.to_csv('object_center.csv')
    
    def load3D(self):
        with open(self.path3D, 'r+b') as f:
            bytefile = f.read()
        encoding = chardet.detect(bytefile)['encoding']
        self.data3D:pd.DataFrame = pd.read_csv(self.path3D, ' ',encoding=encoding, header=None)\
                                    .rename(columns=dict(zip(range(11),'x y z R G B cat conf3D_Crater conf3D_concrete conf3D_background conf3D_UBX'.split())))\
                                    .reindex(columns='x y z R G B cat conf3D_Crater conf3D_background conf3D_UBX'.split())

class Ensembler:
    def __init__(self, data:EnsembleDataManager, thresh=0.5) -> None:
        self.data2D:pd.DataFrame = data.data2D
        self.data3D:pd.DataFrame = data.data3D
        self.cats = data.cats
        self.mean_table = deepcopy(self.data3D)
        self.mean()
        self.apply_thresh(thresh)

    def mean(self):
        self.data2D.sort_values('conf',inplace=True,ascending=True)
        self.mean_table['conf2D_Crater conf2D_UBX'.split()] = 0
        self.mean_table['conf2D_background'] = self.mean_table['conf3D_background']
        for nametuple in self.data2D.itertuples():
            trues = (nametuple.x1<=self.mean_table['x'])&\
                    (self.mean_table['x']<=nametuple.x2)&\
                    (nametuple.y2<=self.mean_table['y'])&\
                    (self.mean_table['y']<=nametuple.y1)
            self.mean_table.loc[trues, f'conf2D_{nametuple.cat}'] = nametuple.conf
        for catId, cat in self.cats.items():
            self.mean_table[f'mean_{cat}'] = self.mean_table[[f'conf2D_{cat}',f'conf3D_{cat}']].mean(axis=1)


    def apply_thresh(self, thresh):
        for catId, cat in self.cats.items():
            is_over:pd.Series = self.mean_table[f'mean_{cat}'] >= thresh
            self.mean_table[f'binary_{cat}'] = is_over.astype(int)
    
    def export_mean(self, export_path:PathLike):
        export_data:pd.DataFrame = self.mean_table['x y z R G B cat mean_Crater conf3D_background mean_UBX'.split()]
        export_data.to_csv(export_path,header=False, index=False, sep=' ')
        print(f'mean 저장 완료')
        
    def export_binary(self, export_path:PathLike):
        export_data:pd.DataFrame = self.mean_table['x y z R G B cat binary_Crater binary_background binary_UBX'.split()]
        export_data.to_csv(export_path,header=False, index=False, sep=' ')
        print(f'binary 저장 완료')

    def export_image(self, export_path, image_size, kind='binary'):
        h, w = image_size
        min_x = self.data3D['x'].min()
        max_x = self.data3D['x'].max()
        min_y = self.data3D['y'].min()
        max_y = self.data3D['y'].max()

        data = deepcopy(self.mean_table[f'x y {kind}_Crater {kind}_background {kind}_UBX'.split()])
        data['x'] = ((data['x']-min_x)/(max_x-min_x)*(w-1)).astype(int)
        data['y'] = ((data['y']-min_y)/(max_y-min_y)*(h-1)).astype(int)

        for catId, cat in self.cats.items():
            data.sort_values(f'{kind}_{cat}', inplace=True)
            droped_data = data.groupby(['x','y']).mean().reset_index()
            # droped_data = data.drop_duplicates(['x','y'],keep='last')
            image = np.zeros(image_size,dtype=np.uint8)
            image[droped_data['y'],droped_data['x']] = (droped_data[f'{kind}_{cat}']*255).astype(int)
            result, encoded_image = cv2.imencode('.jpg',image)

            if result:
                with open(Path(export_path)/f'{kind}_{cat}.jpg', 'w+b') as f:
                    encoded_image.tofile(f)
        print(f'{kind} image 저장 완료')

if __name__=='__main__':
    # camera_spec = {'sensor_size':(8.8,13.2), 'focal_length':8.8, 'distance':50000}#우리거 팬텀
    camera_spec = {'sensor_size':(14.131,10.35), 'focal_length':16, 'distance':50000}#공간정보
    cats = {0:'Crater', 1:'background',2:'UBX'}
    data = EnsembleDataManager(
                'D:/ensemble/images/costum_result_1009_6.json',
                'j:/00.User/K/3D 결과/GPS이동/신_화정관_5_2D_20211009_6회차_Prediction_confidence.txt',
                cats,
                camera_spec,
                79.65,
                'd:/ensemble/images/1009 6.csv')
    ensembler = Ensembler(data, 0.7)
    ensembler.export_binary('D:/ensemble/211214_ensemble_results/1800/binary.csv')
    # ensembler.export_mean('D:/ensemble/211214/mean.csv')
    ensembler.export_image('D:/ensemble/211214_ensemble_results/1800', (1080,1620), kind='conf2D')
    ensembler.export_image('D:/ensemble/211214_ensemble_results/1800', (1080,1620), kind='conf3D')
    ensembler.export_image('D:/ensemble/211214_ensemble_results/1800', (1080,1620), kind='mean')
    ensembler.export_image('D:/ensemble/211214_ensemble_results/1800', (1080,1620), kind='binary')