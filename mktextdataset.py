import os
from sklearn.model_selection import train_test_split
import glob
import shutil
from util import *
from toolz import curry

def readListFromFolder(path, root):
    imglist = []
    for (p, d, f) in os.walk(path):
        imglist_ = filter(lambda i: (i.endswith(".png")|i.endswith(".jpg")), f)
        # imglist_ = filteredimglist(imglist_, p)
        imglist_ = list(map(lambda file: os.path.relpath(os.path.join(p, file), root), imglist_))
        print(f'{p} 폴더에 {len(imglist_)}장의 이미지가 있습니다.')
        imglist.extend(imglist_)
    return imglist

def existAnno(path, root):
    txt = os.path.isfile(os.path.join(root,chgext(path, '.txt')))
    xml = os.path.isfile(os.path.join(root,chgext(path, '.xml')))
    return (xml | txt)

filteredimglist = lambda List, root: list(filter(lambda elm: existAnno(1, root), List))

def readListFromtxt(path, root):
    imglist = txt2list(path)
    imglist = filteredimglist(imglist, root)
    return imglist

def chgelm(List, idx, newvalue):
    List[idx]=newvalue
    return List

def chgMidFd(pathlist, midname):
    allsplitlist = [path.split('/') for path in pathlist]
    # chgsplitlist = [splitpath[1]=midname for splitpath in allsplitlist]
    chgsplitlist = list(map(lambda splitpath: os.path.join(*chgelm(splitpath,1,midname)), allsplitlist))
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

def mkTextDataset(path, test_size=0.2, datasplit=True):
    imglist = []
    allDatasetPath = curry(lambda root: f'{root}/data/original.txt') 
    if os.path.isdir(path):
        root = os.path.normpath(f'{path}/../..')
        imglist = readListFromFolder(path,root)
        allDatasetPath = allDatasetPath(root)
        print(allDatasetPath)
        list2txt(imglist, allDatasetPath)
        print('모든 이미지 경로를 original.txt에 저장했습니다.')

    elif os.path.isfile(path):
        if not os.path.basename(path)=='original.txt':
            print('파일명을 original.txt로 바꿔주세요.')
            return
        root = os.path.normpath(f'{os.path.split(path)[0]}/..')
        imglist = readListFromtxt(path,root)
        allDatasetPath = allDatasetPath(root)
    else:
        print('파일 경로를 확인하세요')
        return
    print(f'총 {len(imglist)}장의 이미지가 있습니다.')
    
    if datasplit:
        trainDatasetPath = os.path.join(root, f'data/train.txt')
        testDatasetPath = os.path.join(root, f'data/test.txt')
    
        trainOrigin, testOrigin = train_test_split(imglist, test_size = test_size, random_state=1)
        trainChgMidFd = chgMidFd(trainOrigin, 'train')
        testChgMidFd = chgMidFd(testOrigin, 'test')

        DatasetBundleSavePath = os.path.join(root, f'data/DatasetBundle.txt')
        DatasetBundle = [trainDatasetPath, testDatasetPath, allDatasetPath]
        list2txt(DatasetBundle, DatasetBundleSavePath)

        list2txt(trainChgMidFd, trainDatasetPath)
        print(f'{len(trainChgMidFd)}장의 train data list를 만들었습니다.')
        trainXmlList = chgext(trainOrigin, '.xml')
        trainTxtList = chgext(trainOrigin, '.txt')
        for img in trainOrigin:
            mkhardlink(os.path.join(root, img), os.path.join(root,'data/train'))
        for xml in trainXmlList:
            xmlwithroot = os.path.join(root, xml)
            if os.path.isfile(xmlwithroot):
                mkhardlink(xmlwithroot, os.path.join(root,'data/train'))
        for txt in trainTxtList:
            txtwithroot = os.path.join(root, txt)
            if os.path.isfile(txtwithroot):
                mkhardlink(txtwithroot, os.path.join(root,'data/train'))


        list2txt(testChgMidFd, testDatasetPath)
        print(f'{len(testChgMidFd)}장의 test data list를 만들었습니다.')
        testXmlList = chgext(testOrigin, '.xml')
        testTxtList = chgext(testOrigin, '.txt')
        for img in testOrigin:
            mkhardlink(os.path.join(root, img), os.path.join(root, 'data/test'))
        for xml in testXmlList:
            xmlwithroot = os.path.join(root, xml)
            if os.path.isfile(xmlwithroot):
                mkhardlink(xmlwithroot, os.path.join(root, 'data/test'))
        for txt in testTxtList:
            txtwithroot = os.path.join(root, txt)
            if os.path.isfile(txtwithroot):
                mkhardlink(txtwithroot, os.path.join(root, 'data/test'))




if __name__ == "__main__":
    # mkTextDataset('/home/tm/Code/darknet/data/train.txt')
    mkTextDataset('/home/tm/Code/darknet/data/data/20200330_2D_Case2',test_size=0.3, datasplit=False)