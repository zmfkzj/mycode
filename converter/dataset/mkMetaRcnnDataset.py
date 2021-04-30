from os.path import *
import sys
sys.path.append(dirname(dirname(dirname(__file__))))

from detection_predict_result import process_gt
import pandas as pd
from util.filecontrol import pickFilename, list2txt
from sklearn.model_selection import train_test_split
import cv2

'''
voc dataset으로부터 Meta R-CNN을 위한 txt 파일 만들기
'''
def mkdataset(filelist, subset, root, novel_class, form='voc'):
    gtpart = process_gt(filelist, subset, root, form=form)
    uniq_class = gtpart['class'].unique()

    #각 class별로 저장
    for val in uniq_class:
        flag = gtpart['class'].map(lambda cls: "1" if cls==val else "-1")
        dataset = (gtpart['img']+" "+flag).unique()
        list2txt(dataset, join(root, f'ImageSets/Main/{val}_{subset}.txt'))

    #train, test class를 나눠 train_first_split.txt 에 저장
    gtpart.set_index('class', inplace=True)
    test_img = gtpart.loc[gtpart.index.intersection(novel_class), 'img'].unique()
    dataset = gtpart.loc[~gtpart['img'].isin(test_img), 'img'].unique()
    list2txt(dataset, join(root, f'ImageSets/Main/train_first_split.txt'))

def train_valid_test_split(gtpart, root):
    #모든 이미지를 임의로 train, val, trainval, test 나눔
    uniq_img = gtpart['img'].unique()
    train_img, testval_img = train_test_split(uniq_img, test_size=0.4, random_state=31)
    val_img, test_img = train_test_split(testval_img, test_size=0.5, random_state=31)
    trainval_img = list(train_img) + list(val_img)

    #save txt
    list2txt(train_img, join(root, f'ImageSets/Main/train.txt'))
    list2txt(val_img, join(root, f'ImageSets/Main/val.txt'))
    list2txt(trainval_img, join(root, f'ImageSets/Main/trainval.txt'))
    list2txt(test_img, join(root, f'ImageSets/Main/test.txt'))

if __name__ == "__main__":
    root = expanduser('c:/Users/mkkim/Desktop/fod/')
    dataset = ['train', 'val', 'trainval', 'test']
    novel_class = [ 'Tag', 'AdjustableClamp', 'Nut', 'BoltNutSet', 'AdjustableWrench', 'MetalSheet', 'Hose' ]


    filelist = join(root, 'ImageSets/Main/default.txt')
    subset = pickFilename(filelist)
    gtpart = process_gt(filelist, subset, root, form='voc')
    train_valid_test_split(gtpart, root)

    for s in dataset:
        filelist = join(root, f'ImageSets/Main/{s}.txt')
        subset = pickFilename(filelist)
        mkdataset(filelist, subset, root, novel_class, form='voc')