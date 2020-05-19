from os.path import basename, splitext
from os import listdir, makedirs
import os
import shutil
'''
labelimg에서 만든 annotation 파일(voc xml 형식)을 cvat에 upload 할 수 있도록 VOC dir tree 로 바꿔줌
'''
allfiles = list(filter(lambda x: x.lower().endswith(('.xml', '.jpg', '.png', '.txt')) , listdir('./')))
newname = map(lambda x: x.replace(' ',''), allfiles)
newname = list(map(lambda x: x.replace(',','_'), newname))

for (old, new) in zip(allfiles, newname):
    print(old, '\t', new)
    os.rename(old, new.encode('utf-8'))

xmlfiles = list(filter(lambda x: x.lower().endswith('.xml') , newname))
folders = ['VOC/ImageSets/Action', 'VOC/ImageSets/Layout', 'VOC/ImageSets/Main', 'VOC/ImageSets/Segmentation', 'VOC/Annotations']
for f in folders:
    makedirs(f, exist_ok=True)
    if 'ImageSets' in f:
        with open(os.path.join(f, 'default.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(map(lambda x: f'{splitext(x)[0]}', xmlfiles)))
    else:
        for xml in xmlfiles:
            shutil.copyfile(xml, os.path.join(f, xml))
