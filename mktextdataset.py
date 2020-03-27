import os
from sklearn.model_selection import train_test_split
import glob
import shutil
from util import *
from toolz.curried import *
from toolz import curry

def readListFromFolder(path, root):
    imglist = []
    for (p, d, f) in os.walk(os.path,join(root, path)):
        imglist_ = filter(lambda i: not (i.endswith(".png")|i.endswith(".jpg")), f)
        imglist_ = filteredimglist(imglist_, p)
        imglist_ = list(map(lambda file: os.path.relpath(os.path.join(p, file), root), f))
        print(f'{p} 폴더에 {len(imglist_)}장의 이미지가 있습니다.')
        imglist.extend(imglist_)
    return imglist

def existAnno(path, root):
    txt = os.path.isfile(os.path.join(root,chgext(path, '.txt')[0]))
    xml = os.path.isfile(os.path.join(root,chgext(path, '.xml')[0]))
    return (xml | txt)

filteredimglist = lambda List, root: list(filter(lambda elm: not existAnno(elm, root), List))

def readListFromtxt(path, root):
    imglist = txt2list(path)
    imglist = filteredimglist(imglist, root)
    return imglist

def chgMidFd(pathlist, midname):
    allsplitlist = [path.split('/') for path in pathlist]
    # chgsplitlist = [splitpath[1]=midname for splitpath in allsplitlist]
    chgsplitlist = list(map(lambda splitpath: splitpath[1] = midname, allsplitlist))
    return chgsplitlist


def mkhardlink(src, link_dir):
    basename = os.path.basename(src)
    link_name = os.path.join(link_dir,basename)
    if not os.path.isdir(link_dir):
        os.makedirs(link_dir)
    try:
        os.link(src,link_name)
    except FileExistsError:
        os.remove(link_name)
        os.link(src,link_name)

def mkTextDataset(path, test_size=0.2):
    imglist = []
    if os.path.isdir(path):
        print('#1')
        root = os.path.normpath(f'{path}/..')
        imglist = readListFromFolder(path,root)
    elif os.path.isfile(path):
        print('#2')
        root = os.path.normpath(f'{os.path.split(path)[0]}/..')
        print(path, root)
        imglist = readListFromtxt(path,root)
    else:
        print('파일 경로를 확인하세요')
        return
    print(f'총 {len(imglist)}장의 이미지가 있습니다.')
    
    train, test = train_test_split(imglist, test_size = test_size, random_state=1)
    train = chgMidFd(train, 'train')
    test = chgMidFd(test, 'test')

    trainDatasetPath = os.path.join(root, f'data/train.txt')
    testDatasetPath = os.path.join(root, f'data/test.txt')
    allDatasetPath = os.path.join(root, f'data/all.txt') 
    
    DatasetBundleSavePath = os.path.join(root, f'data/DatasetBundle.txt')
    DatasetBundle = [trainDatasetPath, testDatasetPath, allDatasetPath]
    list2txt(DatasetBundle, DatasetBundleSavePath)

    list2txt(train, trainDatasetPath)
    print(f'{len(train)}장의 train data list를 만들었습니다.')
    trainXmlList = chgext(train, '.xml')
    trainTxtList = chgext(train, '.txt')
    for img in train:
        mkhardlink(os.path.join(root, img), os.path.join(root,'data/tain'))
    for xml in trainXmlList:
        xmlwithroot = os.path.join(root, xml)
        if os.path.isfile(xmlwithroot):
            mkhardlink(xmlwithroot, os.path.join(root,'data/tain'))
    for txt in trainTxtList:
        txtwithroot = os.path.join(root, txt)
        if os.path.isfile(txtwithroot):
            mkhardlink(txtwithroot, os.path.join(root,'data/tain'))


    list2txt(test, testDatasetPath)
    print(f'{len(test)}장의 test data list를 만들었습니다.')
    testXmlList = chgext(test, '.xml')
    testTxtList = chgext(test, '.txt')
    for img in test:
        mkhardlink(os.path.join(root, img), os.path.join(root, 'data/test'))
    for xml in testXmlList:
        xmlwithroot = os.path.join(root, xml)
        if os.path.isfile(xmlwithroot):
            mkhardlink(xmlwithroot, os.path.join(root, 'data/test'))
    for txt in testTxtList:
        txtwithroot = os.path.join(root, txt)
        if os.path.isfile(txtwithroot):
            mkhardlink(txtwithroot, os.path.join(root, 'data/test'))

    list2txt(train+test, allDatasetPath)
    print('모든 이미지 경로를 all.txt에 저장했습니다.')



if __name__ == "__main__":
    # mkTextDataset('/home/tm/Code/darknet/data/train.txt')
    mkTextDataset('/home/tm/data/train.txt')