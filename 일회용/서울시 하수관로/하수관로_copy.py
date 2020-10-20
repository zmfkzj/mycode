from functools import reduce
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))

from shutil import copy
import xml.etree.ElementTree as ET
from converter.dataset.cvat_merge import load_annotation
from util.filecontrol import folder2list
from tqdm import tqdm
from collections  import defaultdict

'''
cvat에서 분류 후 다운받은 xml 파일에서 tag가 있는 이미지만 골라내어 각 class의 폴더에 복사
'''

def get_taged_images(root:ET.Element) -> list:
    img_tag_dic = defaultdict(list)
    for img in root.iter('image'):
        if (not img.find('tag')==None) & (64000<int(img.attrib['id'])):
            img_tag_dic[osp.basename(img.attrib['name'])].append(img.find('tag').attrib['label'])
    return img_tag_dic

if __name__ == "__main__":
    anno_root = load_annotation('E:\\annotations.xml')
    img_tag_dic = get_taged_images(anno_root)

    capture_path = 'E:\\capture'
    capture_imgs = folder2list(capture_path,['.jpg'])
    captures_dic = dict(map(lambda x: [osp.basename(x), osp.join(capture_path,x)], capture_imgs))

    copy_root = 'E:\\copy'
    for img, tags in tqdm(img_tag_dic.items()):
        copy_paths = list(map(lambda tag: osp.join(copy_root, tag), tags))
        list(map(lambda copy_path: os.makedirs(copy_path, exist_ok=True), copy_paths))
        list(map(lambda copy_path: copy(captures_dic[img], osp.join(copy_path,img)), copy_paths))
    count = sum(map(lambda x: len(x), img_tag_dic.values()))
    print('총 장수: ', count)
