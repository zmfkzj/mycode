import os
from collections.abc import Iterable
from toolz import curry, reduce
import cv2
import numpy as np

def txt2list(txtpath):
    #텍스트 파일의 내용을 리스트로 만들기
    with open(txtpath, 'r') as f:
        filelist = f.readlines()
    filelist = [path.rstrip('\n') for path in filelist]
    filelist = [path for path in filelist if path]
    return filelist

def chgext(path, newext):
    '''
    path의 확장자 변경
    ex) newext = '.xml'
    '''
    ispath = curry(isinstance)(path)
    assert (ispath(str) | ispath(Iterable)), 'path is \'str\' or \'iterable\' type'
    def sub(path, newext):
        other, _ = os.path.splitext(path)
        newpath = f'{other}{newext}'
        return newpath
    
    if isinstance(path,str):
        return sub(path, newext)
    elif isinstance(path, Iterable):
        return [sub(p,newext) for p in path]

def list2txt(List, savepath):
    with open(savepath,'w') as f:
        f.write("\n".join(List))

def ChkNMkFolder(folderpath):
    #folderpath에 폴더 만들기. 중간에 없는 폴더도 만듬
    if not os.path.isdir(folderpath):
        os.makedirs(folderpath)

def pickFilename(path):
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return filename

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def readListFromFolder(root, extlist, path=None):
    #폴더 안에 확장자가 extlist안에 있는 모든 파일 탐색
    filelist = []
    fileter = lambda file, extlist: os.path.splitext(file)[1].lower() in extlist
    if path==None:
        path = root
    for (p, d, f) in os.walk(path):
        filelist_ = filter(lambda i: fileter(i, extlist), f)
        if path!=None:
            filelist_ = list(map(lambda file: os.path.relpath(os.path.join(p, file), root), filelist_))
        print(f'{p}\t폴더에 {len(filelist_)}장의 이미지가 있습니다.')
        filelist.extend(filelist_)
    return filelist



if __name__ == "__main__":
    ChkNMkFolder("/home/tm/Code/mycode/b/c")
