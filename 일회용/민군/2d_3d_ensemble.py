from typing import Dict, List
import pandas as pd
import numpy as np

data3D = pd.DataFrame(columns='R G B x y z class conf'.split())
data2D = pd.DataFrame(columns='x1 y1 x2 y2 class conf'.split())

class Ensemble:
    def __init__(self, data2D:pd.DataFrame, data3D:Dict[str,pd.DataFrame],categories:List[str]) -> None:
        self.data2D = data2D
        self.data3D = data3D
        self.categories= categories

    def mean(self):
        self.data2D.sort_values('conf',inplace=True)
        for cat in self.categories:
            for x1,y1,x2,y2,cat2D,conf in self.data2D.iterrows():
                if cat!=cat2D:
                    continue
                else:
                    zeros3D = pd.Series(np.zeros_like(self.data3D['conf']),name='conf')
                    zeros3D.loc[(x1<=self.data3D['x'])&(self.data3D['x']<=x2)&(y1<self.data3D['y'])&(self.data3D['y']<=y2)&()] = conf
        self.mean_table = self.data3D.copy()
        self.mean_table['conf'] = (self.mean_table['conf']+zeros3D)/2

    def 

