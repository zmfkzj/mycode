from util.drawbbox import *
from mktextdataset import mkTextDataset
from predict_result import *
from tqdm import tqdm
from multiprocessing import Pool
import os
from os.path import join
import shutil

root = '/home/tm/nasrw/터널결함정보/결함정답영상 - 복사본'
newfolder = join(root, 'data/origin')

def mkhardlink(src, link_dir, root):
    relpath = os.path.relpath(src, root)
    link_name = join(link_dir,relpath)
    folder = os.path.split(link_name)[0]
    if not os.path.isdir(folder):
        os.makedirs(folder)
    try:
        os.link(src,link_name)
    except FileExistsError:
        os.remove(link_name)
        os.link(src,link_name)

# if not os.path.isdir(newfolder):
#     os.makedirs(newfolder)

imglist_origin = readListFromFolder(root, ['.jpg', '.png', '.xml', '.txt'])
for f in imglist_origin:
    if not 'data' in f.split('/'):
        mkhardlink(join(root, f),newfolder, root)
        # for img in Origin:
        #         mkhardlink(os.path.join(root, img), os.path.join(root,f'data/{dataset}'))
imglist = mkTextDataset(newfolder, testsize='')

gt_part = process_gt(imglist['all'], 'all', root)

classes = gt_part['class'].unique()
class_count = len(classes)

if not os.path.isdir(os.path.join(root,'data/gt')):
    os.mkdir(os.path.join(root,'data/gt'))

gt_bboxes = gt_part.loc[gt_part['id'].notna(), ['img', 'class', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']].set_index('img')

color = {}
for idx, cls in enumerate(classes):
    color[cls] = (rd.randint(0,256),rd.randint(0,256),rd.randint(0,256))

print(color)

imgs = gt_part['img'].unique()
def run(img_path):
    if img_path[0]!='/':
        ori_img = cv2.imread(os.path.join(root, img_path))
    else:
        ori_img = cv2.imread(img_path)

    fontscale = max(ori_img.shape[:2])/6000
    thick = int(fontscale*4)
    if thick <= 0: thick=1

    img = np.copy(ori_img)
    gt = 'dot'
    save_folder = os.path.join(root,'data/gt')

    try:
        for _, val in gt_bboxes.loc[[img_path],:].iterrows():
            img = drawbbs(img, val[0], val.iloc[1:].tolist(),gt=gt, fontscale=fontscale, thick=thick, color=color[val[0]])
        cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
    except KeyError:
        cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
        pass

with Pool(len(os.sched_getaffinity(0))) as p:
    r = list(tqdm(p.imap(run, imgs), total=len(imgs)))
# run(imgs)
shutil.rmtree(newfolder)