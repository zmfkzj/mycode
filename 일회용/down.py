import requests
import os.path as osp
import os

url = lambda x: f'http://www.ee.oulu.fi/research/imag/WOOD/IMAGES/st{x}.gz'

for i in range(1012, 1851):
    r = requests. get(url(i), allow_redirects=True)
    filename = osp.basename(url(i))

    os.makedirs('wood', exist_ok=True)
    with open(f'wood/{filename}', 'wb') as f:
        f.write(r.content)