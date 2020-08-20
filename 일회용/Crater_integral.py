import pandas as pd
import os.path as osp
import os
import matplotlib.pyplot as plt
import numpy as np
'''
3D 폭파구 부피 구하기
'''
path = osp.expanduser('~/nasrw/mk/폭파구 포인트클라우드/Crater 2.txt')
crater = pd.read_csv(path, header=None, delim_whitespace=True)
crater.columns=['x', 'y', 'z', 'label']
z_ground = crater['z'].quantile(.75)

def sy_z(crater_yz, y):
    z_length = (z_ground - crater_yz.loc[(y<=crater_yz['y']) & (crater_yz['y']<(y+step)), 'z'].median())
    return z_length

V = 0
y_min_global = crater['y'].min()
y_max_global = crater['y'].max()

x_min = crater['x'].min()
x_max = crater['x'].max()
step = (x_max-x_min)/50
x_range = np.arange(x_min, x_max, step=step)
c = 0
for x in x_range:
    crater_yz = crater.loc[(x<=crater['x']) & (crater['x']<(x+step)), ['y', 'z']]
    if crater_yz.size !=0:
        S = 0
        z_length = []
        y_min = crater_yz['y'].min()
        y_max = crater_yz['y'].max()
        y_range =np.arange(y_min, y_max, step=step) 
        for y in y_range:
            crater_z = crater_yz.loc[(y<=crater_yz['y']) & (crater_yz['y']<(y+step)), 'z']
            if len(crater_z) ==0:
                z_length.append(z_length[-1])
                # z_length.append(z_ground - crater_yz['z'].mean())
            else:
                z_length.append(z_ground - crater_z.median())
            S += step*np.abs(z_length[-1])

        plt.clf()
        # crater_yz.sort_values('y').plot.scatter(x='y', y='z', c='cyan')
        plt.plot(y_range, np.repeat(z_ground, len(y_range)), 'r-')
        # plt.plot(y_range, (-np.array(z_length)+z_ground), 'm-')
        plt.bar(y_range, -np.array(z_length), step, z_ground, align='edge', alpha=0.4)
        plt.scatter(crater_yz.sort_values('y')['y'], crater_yz.sort_values('y')['z'])
        plt.ylim(crater['z'].quantile(0.01), crater['z'].quantile(0.99))
        plt.xlim(y_min_global, y_max_global)
        os.makedirs('fig', exist_ok=True)
        plt.savefig(f'fig/fig_{c}_{x}.png', dpi=300)
        c+=1
    else:
        plt.clf()
        plt.ylim(crater['z'].quantile(0.01), crater['z'].quantile(0.99))
        plt.xlim(y_min_global, y_max_global)
        plt.savefig(f'fig/fig_{c}_{x}.png', dpi=300)
        c+=1
    V += step*S

print(V)