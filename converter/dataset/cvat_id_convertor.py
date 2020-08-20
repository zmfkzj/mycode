import xml.etree.ElementTree as ET
import os
import os.path as osp

'''
여러개 task로 분리된 cvat format annotation을 하나로 합치기
'''
def load_annotation(xmlfile:str) -> dict:
    in_file = open(xmlfile)
    tree=ET.parse(in_file)
    root = tree.getroot()
    return root

def id_name(root):
    id_name = dict()
    for img in root.iter('image'):
        id = img.attrib['id']
        file = osp.basename(img.attrib['name']).lower()
        id_name[file] = id
    return id_name

def change_id(before_root, after):
    for img in before_root.iter('image'):
        img.attrib['id'] = after[osp.basename(img.attrib['name'])]
    return before_root


if __name__ == "__main__":
    before_path = osp.expanduser('~/nasrw/mk/datu/1-cvat/annotations.xml')
    after_path = osp.expanduser('~/nasrw/mk/datu/1-cvat/after_annotations.xml')

    before_root = load_annotation(before_path)
    after_root = load_annotation(after_path)
    after_idname = id_name(after_root)

    change_root = change_id(before_root, after_idname)
    tree = ET.ElementTree(change_root)
    save_path = osp.expanduser('~/nasrw/mk/datu/1-cvat/change_annotations.xml')
    tree.write(save_path)

