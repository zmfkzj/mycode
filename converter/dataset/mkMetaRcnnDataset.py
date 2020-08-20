from predict_result import process_gt
import pandas as pd
from util.filecontrol import pickFilename, list2txt
from os.path import *
from sklearn.model_selection import train_test_split
import cv2

'''
voc dataset으로부터 Meta R-CNN을 위한 txt 파일 만들기
'''
def mkdataset(filelist, subset, root, form='voc'):
    gtpart = process_gt(filelist, subset, root, form=form)
    uniq_class = gtpart['class'].unique()
    print(uniq_class)
    gtpart.set_index('class', inplace=True)

    #각 class별로 저장
    for val in uniq_class:
        dataset = gtpart.loc[val, 'img'].map(pickFilename).unique()
        list2txt(dataset, join(root, f'ImageSets/Main/{val}_{subset}.txt'))

    #train, test class를 나눠 train_first_split.txt 에 저장
    # base_class = ['crack', 'efflor', 'desqu', 'peeling', 'leakage'] 
    novel_class = ['Fail', 'MS', 'RE'] 
    test_img = gtpart.loc[novel_class, 'img'].unique()
    dataset = gtpart.loc[~gtpart['img'].isin(test_img), 'img'].map(pickFilename).unique()
    list2txt(dataset, join(root, f'ImageSets/Main/train_first_split.txt'))

    #모든 이미지를 임의로 train, val, trainval, test 나눔
    uniq_img = gtpart['img'].map(pickFilename).unique()
    train_img, testval_img = train_test_split(uniq_img, test_size=0.4, random_state=31)
    val_img, test_img = train_test_split(testval_img, test_size=0.5, random_state=31)
    trainval_img = list(train_img) + list(val_img)
    list2txt(train_img, join(root, f'ImageSets/Main/train.txt'))
    list2txt(val_img, join(root, f'ImageSets/Main/val.txt'))
    list2txt(trainval_img, join(root, f'ImageSets/Main/trainval.txt'))
    list2txt(test_img, join(root, f'ImageSets/Main/test.txt'))

    #mask
    img_size = gtpart.loc[~gtpart["img"].duplicated(), ['img', 'img_H', 'img_W']].copy()
    img_size['img'] = img_size['img'].map(pickFilename)
    img_size = img_size.set_index('img')
    imgs = img_size.index
    size = img_size.to_numpy()
    for (img,shape)  in zip(imgs, size):
        cv2.imwrite(img_size.set_index('img').to_numpy())



if __name__ == "__main__":
    # root = expanduser('~/nasrw/mk/dataset/VOC/VOCdevkit/VOC2007')
    # filelist = join(root, 'ImageSets/Main/shots.txt')
    root = expanduser('~/nasrw/mk/MetaR-CNN/dataset/VOC2007')
    filelist = join(root, 'ImageSets/Main/default.txt')
    subset = pickFilename(filelist)

    # process_gt(filelist, subset, root, form='voc')
    mkdataset(filelist, subset, root, form='voc')

    