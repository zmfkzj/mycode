from os import PathLike
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import chardet
from copy import deepcopy
from pathlib import Path
import cv2
from dtsummary.object import Bbox
from pyproj import Proj, Transformer
from exif import Image

class EnsembleData:
    def __init__(self, path2D:PathLike, path3D:PathLike, cats:Dict[int,str], imagePath) -> None:
        self.path2D = path2D
        self.path3D = path3D
        self.cats = cats
        self.load2D(imagePath)
        self.load3D()
    
    def load2D(self,imagePath):
        with open(self.path2D, 'r+b') as f:
            bytefile = f.read()
        encoding = chardet.detect(bytefile)['encoding']
        with open(self.path2D, 'r', encoding=encoding) as f:
            self.raw2D = json.load(f)
        self.process2D(imagePath)

    def process2D(self, imagePath):
        with open(imagePath,'r+b') as f:
            img = Image(f)
        latitude = np.sum(np.array(img.gps_latitude) / np.array((1,60,3600)))
        longitude = np.sum(np.array(img.gps_longitude) / np.array((1,60,3600)))
        transformer = Transformer.from_crs('epsg:4737','epsg:5186',)
        x,y = transformer.transform(longitude, latitude)
        data = deepcopy(self.raw2D)
        whole_objects = []
        for img in data:
            filename = img['filename']
            objects = img['objects']
            for obj in objects:
                bbox = Bbox(real_size,**obj)
                obj['filename'] = filename
                obj['x1'], obj['y1'], obj['x2'], obj['y2'] = bbox.voc
                whole_objects.append(obj)
        self.data2D = pd.DataFrame([pd.Series(obj) for obj in whole_objects]).rename(columns={'confidence':'conf','name':'cat'})
        self.data2D = self.data2D.reindex(columns='filename x1 y1 x2 y2 conf cat'.split())
    
    def load3D(self):
        with open(self.path3D, 'r+b') as f:
            bytefile = f.read()
        encoding = chardet.detect(bytefile)['encoding']
        self.data3D:pd.DataFrame = pd.read_csv(self.path3D, ' ',encoding=encoding, header=None).rename(columns=dict(zip(range(10),'x y z R G B cat conf3D_Crater conf3D_background conf3D_UBX'.split())))

        #max conf 값을 cat에 따라 할당
        # self.data3D['cat'].replace(dict(zip(range(3),'conf3D_Crater conf3D_background conf3D_UBX'.split())),inplace=True)
        # max_confs = self.data3D['conf3D_Crater conf3D_background conf3D_UBX'.split()].max(axis=1)
        # max_confs = np.expand_dims(max_confs.to_numpy(),1)
        # correct_confs = pd.get_dummies(self.data3D['cat'])*max_confs
        # self.data3D.drop(columns='conf3D_Crater conf3D_background conf3D_UBX'.split(), inplace=True)
        # self.data3D = pd.concat([self.data3D,correct_confs],axis=1)

        self.data3D['filename'] = Path(self.path3D).name
        
        
class Ensembler:
    def __init__(self, data:EnsembleData, thresh=0.5) -> None:
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
            self.mean_table.loc[(nametuple.x1<=self.mean_table['x'])&
                                (self.mean_table['x']<=nametuple.x2)&
                                (nametuple.y1<=self.mean_table['y'])&
                                (self.mean_table['y']<=nametuple.y2),
                                # (self.mean_table['filename']==nametuple.filename),
                                f'conf2D_{nametuple.cat}'] = nametuple.conf
        for catId, cat in self.cats.items():
            self.mean_table[f'mean_{cat}'] = self.mean_table[[f'conf2D_{cat}',f'conf3D_{cat}']].mean(axis=1)


    def apply_thresh(self, thresh):
        for catId, cat in self.cats.items():
            self.mean_table[f'binary_{cat}'] = self.mean_table[f'mean_{cat}'].map(lambda x: 1 if x>=thresh else 0)
    
    def export_mean(self, export_path:PathLike):
        export_data:pd.DataFrame = self.mean_table['x y z R G B cat mean_Crater conf3D_background mean_UBX'.split()]
        export_data.to_csv(export_path,header=False, index=False, sep=' ')
        
    def export_binary(self, export_path:PathLike):
        export_data:pd.DataFrame = self.mean_table['x y z R G B cat binary_Crater binary_background binary_UBX'.split()]
        export_data.to_csv(export_path,header=False, index=False, sep=' ')

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

cats = {0:'Crater', 1:'background',2:'UBX'}
data = EnsembleData('D:/ensemble/custom_dt_result_example.json','D:/ensemble/4-0_0.86_36 sample.txt',cats)
ensembler = Ensembler(data, 0.5)
# ensembler.export_binary('D:/ensemble/binary.csv')
# ensembler.export_mean('D:/ensemble/mean.csv')
# ensembler.export_image('D:/ensemble', (1080,1620), kind='conf2D')
# ensembler.export_image('D:/ensemble', (1080,1620), kind='conf3D')
# ensembler.export_image('D:/ensemble', (1080,1620), kind='mean')
ensembler.export_image('D:/ensemble', (1080,1620), kind='binary')