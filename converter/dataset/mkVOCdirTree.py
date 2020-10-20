from os.path import basename, splitext
from os import listdir, makedirs
import os
import shutil
from collections.abc import Iterable
'''
labelimg에서 만든 annotation 파일(voc xml 형식)을 cvat에 upload 할 수 있도록 VOC dir tree 로 바꿔줌
'''

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

if os.path.isdir(newdir:='new_mkVocDirTree'):
    shutil.rmtree(newdir)
allfiles = folder2list('./', ('.xml', '.jpg', '.png', '.txt'))
allfiles = list(map(lambda path: path.replace('\\', '/'), allfiles))
# allfiles = map(lambda path: os.path.abspath(os.path.join('./',path)), allfiles)
newname = map(lambda x: x.replace(' ',''), allfiles)
newname = map(lambda x: x.replace(',','_'), newname)
# newname = map(lambda x: x.decode('utf-8'), newname)
newname = list(map(lambda x: os.path.join('new_mkVocDirTree',x), newname))

for (old, new) in zip(allfiles, newname):
    print(old, '\t', new)
    # os.rename(old, new.encode('utf-8'))
    os.makedirs(os.path.split(new)[0], exist_ok=True)
    shutil.copyfile(old, new)

# for name in allfiles:
#     os.rename(name, name.encode('utf-8'))

xmlfiles = list(filter(lambda x: x.lower().endswith('.xml') , newname))
# xmlfiles = list(filter(lambda x: x.lower().endswith('.xml') , allfiles))
folders = ['VOC/ImageSets/Action', 'VOC/ImageSets/Layout', 'VOC/ImageSets/Main', 'VOC/ImageSets/Segmentation', 'VOC/Annotations']
folders = map(lambda f: os.path.join(newdir, f), folders)
for f in folders:
    makedirs(f)
    if 'ImageSets' in f:
        with open(os.path.join(f, 'default.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(lambda x: f'{splitext(x)[0]}', xmlfiles)))
    else:
        for xml in xmlfiles:
            makedirs(os.path.join(f, os.path.split(xml)[0]), exist_ok=True)
            shutil.move(xml, os.path.join(f, xml))
            print(xml)
