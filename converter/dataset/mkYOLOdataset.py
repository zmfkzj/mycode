import os
from sklearn.model_selection import train_test_split
import glob
import shutil
from util.filecontrol import *
from toolz import curry, reduce

def existAnno(path, root):
    txt = os.path.isfile(os.path.join(root,chgext(path, '.txt')))
    xml = os.path.isfile(os.path.join(root,chgext(path, '.xml')))
    return (xml | txt)

filteredimglist = lambda List, root: list(filter(lambda elm: existAnno(elm, root), List))

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

def mkTextDataset(path, testsize=0.3, valid=None):
    imglist = []
    allDatasetPath = curry(lambda root: f'{root}/data/original.txt') 

    result = {}
    if os.path.isdir(path):
        root = os.path.normpath(f'{path}/../..')
        imglist = readListFromFolder(root, ['.png', '.jpg'], path)
        result['all'] = imglist
        allDatasetPath = allDatasetPath(root)
        list2txt(imglist, allDatasetPath)
        print('모든 이미지 경로를 original.txt에 저장했습니다.')

    elif os.path.isfile(path):
        if os.path.basename(path) in ['train.txt', 'test.txt', 'valid.txt']:
            print('파일명을 train.txt, test.txtm, valid.txt 외 다른 것으로 바꿔주세요.')
            return
        root = os.path.normpath(f'{os.path.split(path)[0]}/..')
        print(path)
        imglist = readListFromtxt(path,root)
        result['all'] = imglist
        allDatasetPath = allDatasetPath(root)
    else:
        print(f'파일 경로를 확인하세요 {os.path.abspath(path)}')
        return
    print(f'총 {len(imglist)}장의 이미지가 있습니다.')
    
    if testsize:
        trainDatasetPath = os.path.join(root, f'data/train.txt')
        testDatasetPath = os.path.join(root, f'data/test.txt')
    
        trainOrigin, testOrigin = train_test_split(imglist, test_size = testsize, random_state=1)
        trainChgMidFd = chgMidFd(trainOrigin, 'train')
        testChgMidFd = chgMidFd(testOrigin, 'test')

        DatasetBundleSavePath = os.path.join(root, f'data/DatasetBundle.txt')
        DatasetBundle = [trainDatasetPath, testDatasetPath, allDatasetPath]

        if valid:
            validDatasetPath = os.path.join(root, f'data/valid.txt')
            validOrigin, testOrigin = train_test_split(testOrigin, test_size = valid, random_state=1)
            validChgMidFd = chgMidFd(validOrigin, 'valid')
            testChgMidFd = chgMidFd(testOrigin, 'test')
            DatasetBundle = [trainDatasetPath, validDatasetPath, testDatasetPath, allDatasetPath]

        list2txt(DatasetBundle, DatasetBundleSavePath)

        def mk_dataset(Origin, ChgMidFd, DatasetPath, dataset):
            list2txt(ChgMidFd, DatasetPath)
            print(f'{len(ChgMidFd)}장의  {dataset} list를 만들었습니다.')
            XmlList = chgext(Origin, '.xml')
            TxtList = chgext(Origin, '.txt')
            for img in Origin:
                mkhardlink(os.path.join(root, img), os.path.join(root,f'data/{dataset}'))
            for xml in XmlList:
                xmlwithroot = os.path.join(root, xml)
                if os.path.isfile(xmlwithroot):
                    mkhardlink(xmlwithroot, os.path.join(root,f'data/{dataset}'))
            for txt in TxtList:
                txtwithroot = os.path.join(root, txt)
                if os.path.isfile(txtwithroot):
                    mkhardlink(txtwithroot, os.path.join(root,f'data/{dataset}'))

        mk_dataset(trainOrigin, trainChgMidFd, trainDatasetPath, 'train')
        mk_dataset(testOrigin, testChgMidFd, testDatasetPath, 'test')

        result['train'] = trainChgMidFd
        result['test'] = testChgMidFd
        if valid:
            mk_dataset(validOrigin, validChgMidFd, validDatasetPath, 'valid')
            result['valid'] = validChgMidFd

        # list2txt(trainChgMidFd, trainDatasetPath)
        # print(f'{len(trainChgMidFd)}장의 train data list를 만들었습니다.')
        # trainXmlList = chgext(trainOrigin, '.xml')
        # trainTxtList = chgext(trainOrigin, '.txt')
        # for img in trainOrigin:
        #     mkhardlink(os.path.join(root, img), os.path.join(root,'data/train'))
        # for xml in trainXmlList:
        #     xmlwithroot = os.path.join(root, xml)
        #     if os.path.isfile(xmlwithroot):
        #         mkhardlink(xmlwithroot, os.path.join(root,'data/train'))
        # for txt in trainTxtList:
        #     txtwithroot = os.path.join(root, txt)
        #     if os.path.isfile(txtwithroot):
        #         mkhardlink(txtwithroot, os.path.join(root,'data/train'))


        # list2txt(testChgMidFd, testDatasetPath)
        # print(f'{len(testChgMidFd)}장의 test data list를 만들었습니다.')
        # testXmlList = chgext(testOrigin, '.xml')
        # testTxtList = chgext(testOrigin, '.txt')
        # for img in testOrigin:
        #     mkhardlink(os.path.join(root, img), os.path.join(root, 'data/test'))
        # for xml in testXmlList:
        #     xmlwithroot = os.path.join(root, xml)
        #     if os.path.isfile(xmlwithroot):
        #         mkhardlink(xmlwithroot, os.path.join(root, 'data/test'))
        # for txt in testTxtList:
        #     txtwithroot = os.path.join(root, txt)
        #     if os.path.isfile(txtwithroot):
        #         mkhardlink(txtwithroot, os.path.join(root, 'data/test'))
    return result




if __name__ == "__main__":
    # mkTextDataset('/home/tm/Code/darknet/data/train.txt')
    mkTextDataset('/home/tm/nasrw/2D Object Detection_민군 작업 장소/2.트레이닝 사용 이미지/2DOD_민군_20200428_YOLOv4_2/data/original.txt', testsize=0.4, valid=0.5)