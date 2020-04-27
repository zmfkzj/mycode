import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import imageio
import glob
import cv2
from imgaug import augmenters as iaa
import numpy as np
np.random.bit_generator = np.random._bit_generator

from imgaug.augmentables.batches import UnnormalizedBatch
from PIL import Image
import aug_cfg

# ia.seed(1)
Image.MAX_IMAGE_PIXELS = 1000000000

############################################################################
#configure

BATCH_SIZE = 10
goal = 400
inputpath = "/home/tm/nasrw/결함-YOLOv3/cvat upload/task_결함-2020_04_06_05_28_07-yolo/obj_train_data"
suffix = ''
file_preffix = ''
# seqs = [aug_cfg.seq_Crop, aug_cfg.seq_Pad]
seqs = [aug_cfg.seq_CropToFixedsize_Crop]
augmentation = 0
drawbbox = 1
drawlabel = True
clip = False
# downscale = (1400,1400) # tuple, list or None
downscale = None
multiproc = 0
bbox_color = aug_cfg.bbox_color
noob_save = 0 #no object image save

############################################################################
#functions


class InputFileFormatError(Exception):
    def __str__(self):
        return 'inputpath가 폴더나 텍스트파일이 아닙니다.'

def loadlabel(inputpath):
    if os.path.isdir(inputpath):
        dirpath = inputpath
    elif os.path.isfile(inputpath) & inputpath.endswith('.txt'):
        dirpath = os.path.split(inputpath)[0]
    else:
        raise InputFileFormatError
    with open(os.path.join(dirpath,'classes.txt'),'r') as f:
        labels = f.readlines()
    labels = [i.rstrip('\n') for i in labels]
    ids = [str(i) for i in list(range(len(labels)))]
    labeldict = dict(zip(ids, labels))

    return labeldict

labels = loadlabel(inputpath)

def mkpathlist_from_folder(inputpath):
    filelist = os.listdir(inputpath)
    pnglist = [i for i in filelist if (i.endswith(".png")|i.endswith(".jpg"))]
    imgpathlist = [os.path.join(inputpath,i) for i in pnglist]
    return imgpathlist

def yolo2voc(yolocoord, shape):
    '''
    yolo coordinates
    <object-class> <x_center> <y_center> <width> <height>

    VOC coordinates
    x1 = left, x2 = right, y1 = top, y2 = bottom

    shape
    (height, width, channel)
    '''
    x1 = int(shape[1]*(float(yolocoord[1])-float(yolocoord[3])/2))
    x2 = int(shape[1]*(float(yolocoord[1])+float(yolocoord[3])/2))
    y1 = int(shape[0]*(float(yolocoord[2])-float(yolocoord[4])/2))
    y2 = int(shape[0]*(float(yolocoord[2])+float(yolocoord[4])/2))

    return [yolocoord[0], x1, x2, y1, y2]

def voc2yolo(box, size):
    dw = 1./(size[1])
    dh = 1./(size[0])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = np.float32(x*dw)
    w = np.float32(w*dw)
    y = np.float32(y*dh)
    h = np.float32(h*dh)
    return [x,y,w,h]

def drawbbs(image, bbs):
    for i in bbs.bounding_boxes:
        fontscale = max(image.shape[:2])/3000
        thick = int(fontscale*3)
        if thick <= 0: thick=1
        (w,h), _ = cv2.getTextSize(labels[i.label], cv2.FONT_HERSHEY_DUPLEX,fontscale,thick)
        RoundRectangle(image, (i.x1, i.y1), (i.x2, i.y2), bbox_color[i.label], thick)
        # image = i.draw_on_image(image,color=bbox_color[i.label],size=thick)
        if drawlabel:
            text_y = i.y1
            if (i.y1-h)<0:
                text_y = i.y1+h
            cv2.rectangle(image,(i.x1, text_y-h), (i.x1+w,text_y), bbox_color[i.label],-1)
            cv2.rectangle(image,(i.x1, text_y-h), (i.x1+w,text_y), bbox_color[i.label],thick)
            cv2.putText(image, labels[i.label], (i.x1, text_y), cv2.FONT_HERSHEY_DUPLEX,fontscale,(0,0,0),thickness=thick,bottomLeftOrigin=False)
    return image

def merge_dir_ffixes(preffix='', suffix=''):
    new_preffix = preffix
    if augmentation:
        new_preffix = f'aug_{new_preffix}'
    if drawbbox:
        if drawlabel:
            new_preffix = f'{new_preffix}LbBbox_'
        else:
            new_preffix = f'{new_preffix}nLbBbox_'
    if clip:
        new_preffix = f'{new_preffix}Clip_'
    if downscale:
        new_preffix = f'{new_preffix}dScale_'
    return f'{new_preffix}{suffix}'
    
def newdir(imgpath, commonpath):
    identitypath = os.path.relpath(imgpath, commonpath)
    new_folder_preffix = merge_dir_ffixes()
    commondir = os.path.basename(commonpath)
    newsavefolder = f'{new_folder_preffix}{commondir}'
    newsavepath = os.path.join(commonpath,'..',newsavefolder)
    newpath = os.path.normpath(os.path.join(newsavepath, identitypath))
    newpathdir, filename = os.path.split(newpath)
    filename = os.path.splitext(filename)[0]
    return newpathdir, filename

def imgsave(image, imgpath, commonpath, preffix='', suffix=''):
    save_img = np.copy(image)
    if downscale:
        shape = image.shape[:2]
        downscale_index = np.array(downscale)<shape
        if np.any(downscale_index):
            new_shape = np.copy(shape)
            new_shape[downscale_index] = np.array(downscale)[downscale_index]
            new_shape = tuple(new_shape)
            save_img = cv2.resize(image, dsize=new_shape)
            
    newpathdir, filename = newdir(imgpath, commonpath)
    if not os.path.isdir(newpathdir):
        os.makedirs(newpathdir)
    savepath = f'{newpathdir}/{preffix}{filename}{suffix}.jpg'
    imageio.imwrite(savepath, save_img)
    print(f'\tsave - {savepath}')

def rm_zerobbox(bbox_aug):
    xyxy = bbox_aug.to_xyxy_array()
    voc_boxes = []
    for i in range(xyxy.shape[0]):
        # voc_box = voc2yolo(xyxy[i,:], bbox_aug.shape)
        xshape = bbox_aug.shape[1]
        yshape = bbox_aug.shape[0]
        c_x = bbox_aug.bounding_boxes[i].center_x/xshape
        c_y = bbox_aug.bounding_boxes[i].center_y/yshape

        if (c_x<=0) | (c_y<=0) | (c_x>=1) | (c_y>=1):
            continue

        if bbox_aug.bounding_boxes[i].x1<0: bbox_aug.bounding_boxes[i].x1 = 0
        if bbox_aug.bounding_boxes[i].x1>xshape: bbox_aug.bounding_boxes[i].x1 = xshape
        if bbox_aug.bounding_boxes[i].x2<0: bbox_aug.bounding_boxes[i].x2 = 0
        if bbox_aug.bounding_boxes[i].x2>xshape: bbox_aug.bounding_boxes[i].x2 = xshape
        if bbox_aug.bounding_boxes[i].y1<0: bbox_aug.bounding_boxes[i].y1 = 0
        if bbox_aug.bounding_boxes[i].y1>yshape: bbox_aug.bounding_boxes[i].y1 = yshape
        if bbox_aug.bounding_boxes[i].y2<0: bbox_aug.bounding_boxes[i].y2 = 0
        if bbox_aug.bounding_boxes[i].y2>yshape: bbox_aug.bounding_boxes[i].y2 = yshape

        c_x = bbox_aug.bounding_boxes[i].center_x/xshape
        c_y = bbox_aug.bounding_boxes[i].center_y/yshape
        w = bbox_aug.bounding_boxes[i].width/xshape
        h = bbox_aug.bounding_boxes[i].height/yshape
        label =bbox_aug.bounding_boxes[i].label 
        
        voc_box = [label, c_x, c_y, w, h]
        voc_box = [str(ii) for ii in voc_box]
        voc_box = ' '.join(voc_box)
        voc_boxes.append(voc_box)
    return voc_boxes

def bboxsave(voc_boxes, imgpath, commonpath, preffix='', suffix=''):
    newpathdir, filename = newdir(imgpath, commonpath)
    if (not bool(noob_save)) &(not voc_boxes):
        os.remove(f'{newpathdir}/{preffix}{filename}{suffix}.jpg')
        print(f'\tremove - {newpathdir}/{preffix}{filename}{suffix}.jpg')
    else:
        with open(f'{newpathdir}/{preffix}{filename}{suffix}.txt', 'w') as f:
            f.write("\n".join(voc_boxes))

def mkbboxlist(annofile, shape):
    with open(annofile, 'r') as f:
        annlist = f.readlines()
    annlist = [line.rstrip('\n') for line in annlist]
    anncount = len(annlist)
    bboxes = [annlist[i].split(" ") for i in range(anncount)]
    bboxes = [yolo2voc(i, shape) for i in bboxes]
    bboxeslist = [BoundingBox(x1=bboxes[i][1],
                        x2=bboxes[i][2],
                        y1=bboxes[i][3],
                        y2=bboxes[i][4],
                        label=bboxes[i][0]) for i in range(anncount)]
    return bboxeslist

def dotline(img, topleft, bottomright, color, thick):
    a = np.sqrt((bottomright[0]-topleft[0])**2+(bottomright[1]-topleft[1])**2)
    if a==0:
        return
    dotgap = max(img.shape[:2])/50
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

def dotellipse(img, center, r, rotation, start, end, color, thick):
    dr = int((end-start)/4.5)

    start1 = start
    while np.sign(end-start1)==np.sign(dr):
        end1 = start1+dr
        if np.abs(end-start1)< np.abs(dr):
            end1=end
        cv2.ellipse(img, center, r, rotation, start1, end1, color, thick)
        start1 += 2*dr

def RoundRectangle(img, topleft, bottomright, color, line_thickness, linestyle='dot'):
    assert linestyle in ['dot', 'solid'], 'linestyle must be \'dot\' or \'solid\''
    height, width, _ = img.shape
    b_h = int((bottomright[1]-topleft[1])/2)
    b_w = int((bottomright[0]-topleft[0])/2)

    border_radius = int(max([width, height])/30.0)
    r_x = border_radius
    r_y = border_radius
    if border_radius > b_h:
        r_y = b_h
    if border_radius > b_w:
        r_x = b_w

    if linestyle=='solid':
        #draw lines
        #top
        cv2.line(img, topleft, (bottomright[0]-r_x, topleft[1]), color, line_thickness)
        #bottom
        cv2.line(img, (topleft[0]+r_x,bottomright[1]), (bottomright[0]-r_x, bottomright[1]), color, line_thickness)
        #left
        cv2.line(img, topleft, (topleft[0], bottomright[1]-r_y), color, line_thickness)
        #right
        cv2.line(img, (bottomright[0],topleft[1]+r_y), (bottomright[0],bottomright[1]-r_y), color, line_thickness)
        # corners
        # top-right
        cv2.ellipse(img, (bottomright[0]-r_x, topleft[1]+r_y), (r_x, r_y), 0, 0, -90, color, line_thickness)
        #bottom-left
        cv2.ellipse(img, (topleft[0]+r_x, bottomright[1]-r_y), (r_x, r_y), 0, 90, 180, color, line_thickness)
        #bottom-right
        cv2.ellipse(img, (bottomright[0]-r_x, bottomright[1]-r_y), (r_x, r_y), 0, 0, 90, color, line_thickness)

    elif linestyle=='dot':
        #draw lines
        #top
        dotline(img, topleft, (bottomright[0]-r_x, topleft[1]), color, line_thickness)
        #bottom
        dotline(img, (topleft[0]+r_x,bottomright[1]), (bottomright[0]-r_x, bottomright[1]), color, line_thickness)
        #left
        dotline(img, topleft, (topleft[0], bottomright[1]-r_y), color, line_thickness)
        #right
        dotline(img, (bottomright[0],topleft[1]+r_y), (bottomright[0],bottomright[1]-r_y), color, line_thickness)
        #top-right
        dotellipse(img, (bottomright[0]-r_x, topleft[1]+r_y), (r_x, r_y), 0, 0, -90, color, line_thickness)
        #bottom-left
        dotellipse(img, (topleft[0]+r_x, bottomright[1]-r_y), (r_x, r_y), 0, 90, 180, color, line_thickness)
        #bottom-right
        dotellipse(img, (bottomright[0]-r_x, bottomright[1]-r_y), (r_x, r_y), 0, 0, 90, color, line_thickness)

def mkaug(commonpath, filepath):
    image = imageio.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    pathandfilename = os.path.splitext(filepath)[0]
    filename = os.path.basename(pathandfilename)
    errorfile = []

    annotation = f'{pathandfilename}.txt'

    shape = image.shape
    bboxeslist = mkbboxlist(annotation, shape)
    bbs = BoundingBoxesOnImage(bboxeslist, shape=shape)


    if clip:
        for idx, j in enumerate(bbs.bounding_boxes):
            clip_img = j.extract_from_image(image)
            imgsave(clip_img,filepath, commonpath, preffix=labels[j.label], suffix=str(idx))

    if augmentation:
        images = [np.copy(image) for _ in range(BATCH_SIZE)]
        bbses = [bbs.deepcopy() for _ in range(BATCH_SIZE)]
        NB_BATCHES = len(seqs)
        batches = [UnnormalizedBatch(images=images, bounding_boxes=bbses) for _ in range(NB_BATCHES)]
        seq = [i(downscale,shape) for i in seqs]
        c = 0
        try:
            while BATCH_SIZE*NB_BATCHES*c < goal:
                aug = []
                for aug_seq, nb in zip(seq, range(NB_BATCHES)):
                    aug.extend(list(aug_seq.augment_batches(batches[nb], background=multiproc)))
                for idx, x in enumerate(aug):
                    for y in range(len(x.images_aug)):
                        new_file_preffix = f'{file_preffix}'
                        number =BATCH_SIZE*NB_BATCHES*c + idx*BATCH_SIZE + y + 1
                        new_suffix = f'{suffix}-{str(number)}'
                        voc_boxes = rm_zerobbox(x.bounding_boxes_aug[y])
                        if (not voc_boxes) & (not noob_save):
                            print(f'\tpass - {number}번째 aug_image에 Bounding box가 없음. no object image save off')
                            continue
                        if (not voc_boxes) & noob_save:
                            new_file_preffix = f'noob_{new_file_preffix}'
                        img = x.images_aug[y]
                        if drawbbox:
                            img = drawbbs(img, x.bounding_boxes_aug[y])
                        imgsave(img, filepath, commonpath, preffix=new_file_preffix, suffix=new_suffix)
                        bboxsave(voc_boxes, filepath, commonpath, preffix=new_file_preffix, suffix=new_suffix)
                c+=1
        except:
            print('Memory Error')
            errorfile.append(filepath)
            pass
        with open(os.path.join(commonpath,'log.txt'),'a') as f:
            f.write('\n')
            f.write('\n'.join(errorfile))
    else:
        if drawbbox:
            image = drawbbs(image, bbs)
        imgsave(image, filepath, commonpath)
        print('{} is saved\n'.format(filename))

def load_imgpathlist(inputpath):
    if os.path.isdir(inputpath):
        imgpathlist = mkpathlist_from_folder(inputpath)
    elif os.path.splitext(inputpath)[1]=='.txt':
        with open(inputpath, 'r') as f:
            imgpathlist = f.readlines()
        imgpathlist = [i.rstrip('\n') for i in imgpathlist]
    else:
        raise InputFileFormatError
    return imgpathlist


######################################################################

if __name__ == "__main__":
    imgpathlist = load_imgpathlist(inputpath)
    totalconunt = len(imgpathlist)
    commonpath = os.path.commonpath(imgpathlist)
    while imgpathlist:
        for i, path in enumerate(imgpathlist):
            # if i==1:
            #     break
            print('{}\t/ {}\tcurrent image : {}'.format(i+1, totalconunt,os.path.basename(path)))
            mkaug(commonpath, path)
        imgpathlist = load_imgpathlist(f'{commonpath}/log.txt')
        os.remove(f'{commonpath}/log.txt')
        BATCH_SIZE = int(BATCH_SIZE*0.7)

    # imgpathlist_log = load_imgpathlist(f'{commonpath}/log.txt')
    # while imgpathlist_log:
    #     totalconunt = len(imgpathlist_log)
    #     for i, path in enumerate(imgpathlist_log):
    #         # if i==1:
    #         #     break
    #         print('{}\t/ {}\tcurrent image : {}'.format(i+1, totalconunt,os.path.basename(path)))
    #         mkaug(commonpath, path)
    #     imgpathlist_log = load_imgpathlist(f'{commonpath}/log.txt')
    #     os.remove(f'{commonpath}/log.txt')