import cv2
import numpy as np
import pandas as pd
import os
import random as rd
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from multiprocessing import Pool
import asyncio as aio
from filecontrol import pickFilename
import sys

sys.path.append(os.path.expanduser("~/host/nasrw/mk/mycode"))
sys.path.append(os.path.expanduser("~/nasrw/mk/mycode"))
from converter.annotation.voc2yolo import calxyWH

rd.seed(31)

def drawbbs(img, bbox, gt=False, thick=1, color=(255,0,0)):
    '''
    bbox = (left, top, right, bottom)
    '''
    assert gt in [False, 'dot', 'solid'], 'gt must be False:bool,\'dot\' or \'solid\''
    bbox = list(map(lambda coord: int(np.around(coord)),bbox))
    if gt:
        RoundRectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick, linestyle=gt)
    else:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

    return img

def drawlabel(img, label, bbox, fontscale=1, thick=1, color=(255,0,0)):
    bbox = list(map(lambda coord: int(np.around(coord)),bbox))
    #draw label
    fontpath = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"     
    font = ImageFont.truetype(fontpath, int(fontscale))
    img_label = Image.new('RGB', (int(fontscale*100),int(fontscale*1.5)))
    draw = ImageDraw.Draw(img_label)
    w, h = draw.textsize(label, font=font)

    draw.rectangle((0, 0, w-1, int(h*1.1)), fill=color, width=-1)
    draw.rectangle((0, 0, w-1, int(h*1.1)), fill=color, width=thick)
    draw.text((0, 0), label, font = font, fill = (1,1,1))
    img_label = np.array(img_label)

    text_y = bbox[1] - h #label 위치 조정
    if text_y<0:
        text_y=0
    img_shape = img.shape
    label_idx = np.where(img_label>0)
    label_idx_inImg = [label_idx[0]+text_y, label_idx[1]+bbox[0], label_idx[2]]
    for idx, val in enumerate(label_idx_inImg[:2]):
        out_range = np.where(val>=img_shape[idx])
        label_idx = [np.delete(i, out_range) for i in label_idx]
        label_idx_inImg = [np.delete(i, out_range) for i in label_idx_inImg]
    img[tuple(label_idx_inImg)] = img_label[tuple(label_idx)]

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
    root = os.path.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_20200622_YOLOv2_1/')
    form = 'yolo'

    #perObj cvs
    csv_path = os.path.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_20200622_YOLOv2_1/test2_perObj_yolov2-voc_best_It0.5_ct0.25_200627-1215.csv')
    subset = pickFilename(csv_path).split('_')[0]
    perobj = pd.read_csv(csv_path, encoding='euc-kr')
    df_class = perobj['class'].unique()
    class_count = len(df_class)

    if not os.path.isdir(os.path.join(root,f"{subset}_gt")):
        os.makedirs(os.path.join(root,f"{subset}_gt"))
    if not os.path.isdir(os.path.join(root,f"{subset}_pred")):
        os.makedirs(os.path.join(root,f"{subset}_pred"))

    gt_bboxes = perobj.loc[perobj['id'].notna(), ['img', 'class', 'gt_left', 'gt_top', 'gt_right', 'gt_bottom']].set_index('img')
    pred_bboxes = perobj.loc[perobj['conf'].notna(), ['img', 'class', 'pred_left', 'pred_top', 'pred_right', 'pred_bottom']].set_index('img')

    color = {}
    for idx, cls in enumerate(df_class):
        color[cls] = (rd.randint(0,256),rd.randint(0,256),rd.randint(0,256))

    print(color)

    imgs = perobj['img'].unique()
    async def run(img_in_df):
        loop = aio.get_event_loop()
        if form=='yolo':
            img_path = os.path.join(root,'..',img_in_df)
        elif form=='voc':
            img_path = os.path.join(root, img_in_df)
        else:
            raise
        ori_img = await loop.run_in_executor(None, cv2.imread,img_path)

        fontscale = max(ori_img.shape[:2])*(10/250)
        thick = int(max(ori_img.shape[:2])*(1/250))
        if thick <= 0: thick=1

        for bboxes in [gt_bboxes, pred_bboxes]:
            img = np.copy(ori_img)
            if id(bboxes)==id(gt_bboxes):
                gt = 'dot'
                save_folder = os.path.join(root, f"{subset}_gt")
            else:
                gt = False
                save_folder = os.path.join(root,f"{subset}_pred")

            try:
                for _, val in bboxes.loc[[img_in_df],:].iterrows():
                    img = drawbbs(img,val.iloc[1:].tolist(), gt=gt, thick=thick, color=color[val[0]])
                    if val[0]=='Crater':
                        img = drawlabel(img, f'{val[0]} S={0.7854*(val.iloc[3]-val.iloc[1])*(val.iloc[4]-val.iloc[2])}', val.iloc[1:].tolist(), fontscale=fontscale, thick=thick, color=color[val[0]])
                        #bounding box의 중심과 가로세로 크기 계산
                        xyWH = tuple((calxyWH(val.iloc[1:].tolist(), img.shape[:2]) * np.tile(img.shape[:2][::-1], 2)).astype(int))
                        #타원 그리기
                        overlay = np.zeros(img.shape)
                        overlay = cv2.ellipse(overlay, xyWH[:2], tuple(map(lambda x: x//2, xyWH[2:])), 0, 0, 360, color[val[0]], thickness=-1)
                        #겹치기
                        pos = np.where(overlay>0)
                        img[pos] = (img[pos]*0.5 + overlay[pos]*0.5).astype(np.int8)
                    else:
                        img = drawlabel(img, val[0], val.iloc[1:].tolist(), fontscale=fontscale, thick=thick, color=color[val[0]])


                await loop.run_in_executor(None, cv2.imwrite,f'{save_folder}/{os.path.basename(img_path)}', img)
            except KeyError as e:
                # cv2.imwrite(f'{save_folder}/{os.path.basename(img_path)}', img)
                pass

    # with Pool() as p:
    #     r = list(tqdm(p.imap(run, imgs), total=len(imgs)))

    async def exe():
        tasks=[]
        for img in tqdm(imgs):
            tasks.append(run(img))
        await aio.gather(*tasks)

    aio.run(exe())
