from toolz import *
from toolz.curried import *
from util import *
from argparse import ArgumentParser
import sys
import os
import shutil
from tqdm import tqdm


def joinPath(outputPath):
    VOCdirTree = ["SegmentationObject", "SegmentationClass", "JPEGImages", "ImageSets/Layout", "ImageSets/Main","ImageSets/Segmentation","Annotations"]
    joinDirs = list(map(lambda dir: os.path.join(args.outputPath,f'VOC{args.year}',dir), VOCdirTree))
    return joinDirs

def mkVOCdir(joinDirs):
    for folderName in joinDirs:
        ChkNMkFolder(folderName)

def cpfiles(imgPathList, joinDirs):
    annoPathList = chgext(imgPathList,".xml")

    for imgs in tqdm(imgPathList, desc="copy images"):
        shutil.copy(imgs, f'{joinDirs[2]}/')
    for annos in tqdm(annoPathList, desc="copy annotations"):
        shutil.copy(annos, f'{joinDirs[6]}/')

def saveDatasetBundle(bundlePath, joinDirs):
    DatasetBundle = txt2list(bundlePath)
    DatasetBasename = map(os.path.basename, DatasetBundle)
    DatasetBundle_release = map(txt2list,DatasetBundle)
    bundleFilename = [list(map(pickFilename,Dataset)) for Dataset in DatasetBundle_release]
    for (List, basename) in zip(bundleFilename, DatasetBasename):
        list2txt(List, os.path.join(joinDirs[4],basename))
        print(f"save\t{basename}")


if __name__ == "__main__":
    #option
    parser = ArgumentParser(description="이미지의 경로들의 list를 입력받아 VOC dataset 형식으로 변환")
    parser.add_argument("inputPath", type=str, help="이미지의 경로들이 적힌 txt 파일의 경로")
    parser.add_argument("bundlePath", type=str, help="몇년도로 폴더를 만들건지")
    parser.add_argument("--outputPath", type=str, default="./", help="변환한 VOC dataset이 위치할 경로")
    parser.add_argument("--year", type=int, default=2007, help="몇년도로 폴더를 만들건지")

    args = parser.parse_args()

    joinDirs = joinPath(args.outputPath)
    mkVOCdir(joinDirs)
    
    ImgPathListFromYolotype = txt2list(args.inputPath)
    # cpfiles(ImgPathListFromYolotype, joinDirs)
    saveDatasetBundle(args.bundlePath, joinDirs)