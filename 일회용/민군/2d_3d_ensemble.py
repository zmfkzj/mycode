from math import exp
from os import PathLike
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import chardet
from copy import deepcopy
from pathlib import Path
import cv2


class EnsembleData:
    def __init__(self, path2D:PathLike, path3D:PathLike, cats:Dict[int,str]) -> None:
        self.path2D = path2D
        self.path3D = path3D
        self.cats = cats
        self.load2D()
        self.load3D()
    
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
            for obj in objects:
                obj['filename'] = filename
                obj['x1'], obj['y1'], obj['x2'], obj['y2'] = obj['voc_bbox']
                whole_objects.append(obj)
        self.data2D = pd.DataFrame([pd.Series(obj) for obj in whole_objects]).drop(columns='voc_bbox').rename(columns={'confidence':'conf','name':'cat'})
    
    def load3D(self):
        with open(self.path3D, 'r+b') as f:
            bytefile = f.read()
        encoding = chardet.detect(bytefile)['encoding']
        self.data3D = pd.read_csv(self.path3D, ' ',encoding=encoding, header=None).rename(columns=dict(zip(range(10),'x y z R G B cat conf3D_Crater conf3D_background conf3D_UBX'.split())))
        self.data3D['filename'] = Path(self.path3D).name
        
        
class Ensembler:
    def __init__(self, data:EnsembleData) -> None:
        self.data2D:pd.DataFrame = data.data2D
        self.data3D:pd.DataFrame = data.data3D
        self.cats = data.cats
        self.mean_table = deepcopy(self.data3D)
        self.mean()
        self.apply_thresh(0.5)

    def mean(self):
        self.data2D.sort_values('conf',inplace=True)
        self.mean_table['conf2D_Crater conf2D_UBX'.split()] = 0
        self.mean_table['conf2D_background'] = self.mean_table['conf3D_background']
        for nametuple in self.data2D.itertuples():
            self.mean_table.loc[(nametuple.x1<=self.data3D['x'])&
                                (self.data3D['x']<=nametuple.x2)&
                                (nametuple.y1<=self.data3D['y'])&
                                (self.data3D['y']<=nametuple.y2)&
                                (self.data3D['filename']==nametuple.filename),
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

        print(data.sum())

        for catId, cat in self.cats.items():
            data.sort_values(f'{kind}_{cat}', inplace=True)
            droped_data = data.drop_duplicates(['x','y'],keep='last')
            image = np.zeros(image_size,dtype=np.int8)
            image[droped_data['y'],droped_data['x']] = (droped_data[f'{kind}_{cat}']*255).astype(int)
            result, encoded_image = cv2.imencode('.jpg',image)

            if result:
                with open(Path(export_path)/f'{kind}_{cat}.jpg', 'w+b') as f:
                    encoded_image.tofile(f)

cats = {0:'Crater', 1:'background',2:'UBX'}
data = EnsembleData('D:/ensemble/custom_dt_result_example.json','D:/ensemble/4-0_0.86_36 sample.txt',cats)
ensembler = Ensembler(data)
# ensembler.export_binary('D:/ensemble/binary.csv')
# ensembler.export_mean('D:/ensemble/mean.csv')
ensembler.export_image('D:/ensemble', (1080,1620), kind='conf2D')
ensembler.export_image('D:/ensemble', (1080,1620), kind='conf3D')
ensembler.export_image('D:/ensemble', (1080,1620), kind='mean')
print()