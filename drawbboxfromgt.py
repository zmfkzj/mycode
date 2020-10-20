from util.drawbbox import *
from converter.dataset.mkYOLOdataset import mkTextDataset
from detection_predict_result import *
from tqdm import tqdm
from multiprocessing import Pool, freeze_support
import os
from os.path import join
import shutil
import platform
from PIL import Image

root = os.path.expanduser('Y:\\디지털재단 데이터\\검수용 이미지\\test\\test2')
newfolder = join(root, 'draw_data/origin')
# newfolder = join(root)

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

if os.path.isdir(newfolder):
    shutil.rmtree(join(newfolder,'..'))

imglist_origin = folder2list(root, ['.jpg', '.png', '.xml', '.txt'])
for f in imglist_origin:
    if not 'draw_data/gt' in f:
        mkhardlink(join(root, f),newfolder, root)
        # for img in Origin:
        #         mkhardlink(os.path.join(root, img), os.path.join(root,f'data/{dataset}'))
imglist = mkTextDataset(newfolder, testvalsize=None)

gt_part = process_gt(imglist['all'], 'all', join(root, 'draw_data'), form='voc')
# gt_part = pd.read_csv(os.path.expanduser('~/nasrw/mk/MetaR-CNN/dataset/VOC2007/gtpart_default.csv'), encoding='euc-kr')

classes = gt_part['class'].unique()
class_count = len(classes)

if not os.path.isdir(os.path.join(root,'draw_data/gt')):
    os.makedirs(os.path.join(root,'draw_data/gt'))

gt_bboxes = gt_part.loc[gt_part['id'].notna(), ['img', 'class', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']].set_index('img')

labelmap = txt2list(join(root, 'labelmap.txt'))
color = {}
for line in labelmap:
    if line.strip(' ')[0] == '#':
        continue
    else:
        split_line = line.split(':')
        if split_line[0] == 'background':
            continue
        else:
            color[split_line[0]] = tuple([int(x) for x in split_line[1].split(',')])
# color = {}
# for idx, cls in enumerate(classes):
#     color[cls] = (rd.randint(0,256),rd.randint(0,256),rd.randint(0,256))

print(color)

imgs = gt_part['img'].unique()
def run(img_path):
    if os.path.isabs(img_path):
        # ori_img = cv2.imread(img_path)
        ori_img = np.array(Image.open(img_path))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
    else:
        # ori_img = cv2.imread(os.path.join(root, img_path))
        ori_img = np.array(Image.open(os.path.join(root, img_path)))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

    fontscale = max(ori_img.shape[:2])*(10/250)
    thick = int(max(ori_img.shape[:2])*(1/250))
    if thick <= 0: thick=1

    img = np.copy(ori_img)
    gt = 'dot'
    save_folder = os.path.join(root,'draw_data/gt')

    try:
        for _, val in gt_bboxes.loc[[img_path],:].iterrows():
            img = drawbbs(img, val.iloc[1:].tolist(),gt=gt, thick=thick, color=color[val[0]])
            img = drawlabel(img, val[0], val.iloc[1:].tolist(),fontscale=fontscale, thick=thick, color=color[val[0]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        save_path = f'{save_folder}/{img_path[12:]}'
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        Image.fromarray(img).save(save_path, format='JPEG')
        # cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
    except KeyError:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        save_path = f'{save_folder}/{img_path[12:]}'
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        Image.fromarray(img).save(save_path, format='JPEG')
        # cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
        pass

osname = platform.system()
if osname == 'Windows':
    list(tqdm(map(run, imgs)))
else:
    with Pool() as p:
        r = list(tqdm(p.imap(run, imgs), total=len(imgs)))
# run(imgs)
shutil.rmtree(newfolder)