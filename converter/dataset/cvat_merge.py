import xml.etree.ElementTree as ET
import os
import os.path as osp
from glob import glob

'''
여러개 task로 분리된 cvat format annotation을 하나로 합치기
'''
def load_annotation(xmlfile:str) -> ET.Element:
    in_file = open(xmlfile, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    return root

def id_name(root:ET.Element)->dict:
    id_name = dict()
    for img in root.iter('image'):
        id = img.attrib['id']
        file = osp.basename(img.attrib['name']).lower()
        id_name[file] = id
    return id_name

def change_id(before_root:ET.Element, after:dict) -> ET.Element:
    for img in before_root.iter('image'):
        img.attrib['id'] = after[osp.basename(img.attrib['name']).lower()]
    return before_root
    
def get_images(root:ET.Element) -> list:
    images = []
    for img in root.iter('image'):
        images.append(img)
    return images

def create_root() -> ET.Element:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    ET.SubElement(root, "meta").text = ''
    return root

def indent(elem, level=0): #자료 출처 https://goo.gl/J8VoDK
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

if __name__ == "__main__":
    #합칠 cvat format의 xml 파일들이 있는 폴더 경로
    label_files = glob(osp.expanduser('~/host/nasrw/mk/datu/sq_defect_cvat-form/*/*.xml'))
    #annotation이 없이 이미지만 합쳐서 업로드한 task에서 다운받은 cvat format의 xml 파일
    empty_new_path = osp.expanduser('~/host/nasrw/mk/datu/1-cvat/after_annotations.xml')

    images = []
    for path in label_files:
        root = load_annotation(osp.expanduser(path))
        images.extend(get_images(root))
    
    merge_root = create_root()
    merge_root.extend(images)

    empty_new_root = load_annotation(empty_new_path)
    empty_nw_idname = id_name(empty_new_root)

    change_root = change_id(merge_root, empty_nw_idname)
    indent(change_root)
    tree = ET.ElementTree(change_root)
    save_path = osp.expanduser('~/host/nasrw/mk/datu/1-cvat/change_annotations.xml')
    tree.write(save_path, encoding='utf-8')
        

