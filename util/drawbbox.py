import cv2
import numpy as np
import pandas as pd
import os
import random as rd
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from multiprocessing import Pool

rd.seed(31)

def drawbbs(img, label, bbox, gt=False, fontscale=1, thick=1, color=(255,0,0)):
    '''
    bbox = (left, top, right, bottom)
    '''
    assert gt in [False, 'dot', 'solid'], 'gt must be False:bool,\'dot\' or \'solid\''
    bbox = list(map(int,bbox))
    if gt:
        RoundRectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick, linestyle=gt)
    else:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

    #draw label
    fontpath = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"     
    font = ImageFont.truetype(fontpath, int(30*fontscale))
    img_pil = Image.new('RGB', (int(600+fontscale),int(35*fontscale)))
    draw = ImageDraw.Draw(img_pil)
    w, h = draw.textsize(label, font=font)

    text_y = bbox[1] - h #label 위치 조정
    if text_y<0:
        text_y=0
    draw.rectangle((0, 0, w-1, h-1), fill=color, width=-1)
    draw.rectangle((0, 0, w-1, h-1), fill=color, width=thick)
    draw.text((0, 0), label, font = font, fill = (1,1,1))
    img_pil = np.array(img_pil)

    img_shape = img.shape
    label_idx = np.where(img_pil>0)
    label_idx_inImg = [label_idx[0]+text_y, label_idx[1]+bbox[0], label_idx[2]]
    for idx, val in enumerate(label_idx_inImg[:2]):
        out_range = np.where(val>=img_shape[idx])
        label_idx = [np.delete(i, out_range) for i in label_idx]
        label_idx_inImg = [np.delete(i, out_range) for i in label_idx_inImg]

    img[tuple(label_idx_inImg)] = img_pil[tuple(label_idx)]
    
    return img

def RoundRectangle(img, topleft, bottomright, color, thick, linestyle='dot'):
    assert linestyle in ['dot', 'solid'], 'linestyle must be \'dot\' or \'solid\''
    height, width, _ = img.shape
    b_h = int((bottomright[1]-topleft[1])/2)
    b_w = int((bottomright[0]-topleft[0])/2)

    border_radius = thick*20
    r_x = border_radius
    r_y = border_radius
    if border_radius > b_h:
        r_y = b_h
    if border_radius > b_w:
        r_x = b_w

    if linestyle=='solid':
        drline = cv2.line
        drellipsis = cv2.ellipse
    elif linestyle=='dot':
        drline = dotline
        drellipsis = dotellipse

    drline(img, topleft, (bottomright[0]-r_x, topleft[1]), color, thick)#top
    drline(img, (topleft[0]+r_x,bottomright[1]), (bottomright[0]-r_x, bottomright[1]), color, thick)#bottom
    drline(img, topleft, (topleft[0], bottomright[1]-r_y), color, thick)#left
    drline(img, (bottomright[0],topleft[1]+r_y), (bottomright[0],bottomright[1]-r_y), color, thick)#right
    drellipsis(img, (bottomright[0]-r_x, topleft[1]+r_y), (r_x, r_y), 0, 0, -90, color, thick)#top-right
    drellipsis(img, (topleft[0]+r_x, bottomright[1]-r_y), (r_x, r_y), 0, 90, 180, color, thick)#bottom-left
    drellipsis(img, (bottomright[0]-r_x, bottomright[1]-r_y), (r_x, r_y), 0, 0, 90, color, thick)#bottom-right

def dotellipse(img, center, r, rotation, start, end, color, thick):
    dr = int((end-start)/4.5)

    start1 = start
    while np.sign(end-start1)==np.sign(dr):
        end1 = start1+dr
        if np.abs(end-start1)< np.abs(dr):
            end1=end
        cv2.ellipse(img, center, r, rotation, start1, end1, color, thick)
        start1 += 2*dr

def dotline(img, topleft, bottomright, color, thick):
    a = np.sqrt((bottomright[0]-topleft[0])**2+(bottomright[1]-topleft[1])**2)
    if a==0:
        return
    dotgap = thick*10
    b = a/dotgap
    dx = int((bottomright[0]-topleft[0])/b)
    dy = int((bottomright[1]-topleft[1])/b)

    x1, y1 = topleft
    while (np.sign(bottomright[0]-x1)==np.sign(dx)) & (np.sign(bottomright[1]-y1)==np.sign(dy)):
        end_x = x1+dx
        end_y = y1+dy

        if np.abs(bottomright[0]-end_x)<np.abs(dx):
            end_x = bottomright[0]
        if np.abs(bottomright[1]-end_y)<np.abs(dy):
            end_y = bottomright[1]
            
        cv2.line(img, (x1, y1), (end_x, end_y), color, thick)
        x1 += 2*dx
        y1 += 2*dy

if __name__ == "__main__":
    # root = '/home/tm/nasrw/2D Object Detection_민군 작업 장소/2.트레이닝 사용 이미지/2DOD_민군_20200504_YOLOv4_1'
    root = ''
    gt_part = pd.read_csv('data/train_perObj_Crack_8_yolov3-voc_panorama_final_It0.5_ct0.25_200507-1755.csv', encoding='euc-kr')
    df_class = gt_part['class'].unique()
    class_count = len(df_class)

    if not os.path.isdir(os.path.join(root,'data/gt')):
        os.mkdir(os.path.join(root,'data/gt'))
    if not os.path.isdir(os.path.join(root,'data/pred')):
        os.mkdir(os.path.join(root,'data/pred'))

    gt_bboxes = gt_part.loc[gt_part['id'].notna(), ['img', 'class', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']].set_index('img')
    pred_bboxes = gt_part.loc[gt_part['conf'].notna(), ['img', 'class', 'pred_left', 'pred_top', 'pred_right', 'pred_bottom']].set_index('img')

    color = {}
    for idx, cls in enumerate(df_class):
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

        for bboxes in [gt_bboxes, pred_bboxes]:
            img = np.copy(ori_img)
            if id(bboxes)==id(gt_bboxes):
                gt = 'dot'
                save_folder = os.path.join(root,'data/gt')
            else:
                gt = False
                save_folder = os.path.join(root,'data/pred')

            try:
                for _, val in bboxes.loc[[img_path],:].iterrows():
                    img = drawbbs(img, val[0], val.iloc[1:].tolist(),gt=gt, fontscale=fontscale, thick=thick, color=color[val[0]])
                cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
            except KeyError:
                # cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
                pass

    with Pool(16) as p:
        r = list(tqdm(p.imap(run, imgs), total=len(imgs)))