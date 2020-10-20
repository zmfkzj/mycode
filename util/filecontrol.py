# -*- coding:utf-8 -*-
import os
from collections.abc import Iterable
from toolz import curry, reduce
import cv2
import numpy as np
from typing import *

def txt2list(txtpath):
    #텍스트 파일의 내용을 리스트로 만들기
    with open(txtpath, 'r', encoding='utf-8') as f:
        filelist = f.readlines()
    filelist = [path.rstrip('\n') for path in filelist]
    filelist = [path for path in filelist if path]
    return filelist

def folder2list(root:str, extlist:Iterable, path=None) -> list:
    '''
    폴더 및 하위 폴더 안에 확장자가 extlist안에 있는 모든 파일 탐색
    예시)
    extlist = ['.xml', '.jpg']
    '''
    list(map(lambda ext: ext.lower(), extlist))
    filelist = []
    isext = lambda file, extlist: os.path.splitext(file)[1].lower() in extlist
    if path==None:
        path = root
    for (path, dir, file) in os.walk(path):
        filelist_ = filter(lambda i: isext(i, extlist), file)
        if path!=None:
            filelist_ = list(map(lambda file: os.path.relpath(os.path.join(path, file), root), filelist_))
        print(f'{path}\t폴더에 {len(filelist_)}장의 파일이 있습니다.')
        filelist.extend(filelist_)
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

def chgfilename(path:str, newfilename:str):
    file, ext = os.path.splitext(path)
    root, _ = os.path.split(file)
    newpath = os.path.join(root, f'{newfilename}{ext}')
    return newpath

def addsuffix(path:str, suffix:Union[str, List[str]]):
    file, ext = os.path.splitext(path)
    root, filename = os.path.split(file)
    newfilename = [filename]
    if isinstance(suffix, str):
        newfilename.append(suffix)
    else:
        newfilename.extend(suffix)
    newfilename = '_'.join(newfilename)
    newpath = os.path.join(root, f'{newfilename}{ext}')
    return newpath

def list2txt(List, savepath, mode='w'):
    with open(savepath,mode) as f:
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




if __name__ == "__main__":
    ChkNMkFolder("/home/tm/Code/mycode/b/c")
