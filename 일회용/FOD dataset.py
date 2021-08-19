import os
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path
from collections import defaultdict

anno_dir_list = []
img_dir_list = []
for p, ds, _ in os.walk('./FullDataset/Annotations/'):
    paths = [Path(p)/d for d in ds if d == 'Annotations']
    anno_dir_list.extend(paths)
    paths = [Path(p)/d for d in ds if d == 'frame']
    img_dir_list.extend(paths)

# for img_dir in img_dir_list:
#     shutil.rmtree(img_dir)

for anno_dir in anno_dir_list:
    os.rename(anno_dir, anno_dir.with_name('frame'))

xml_list = []
for p, _, fs in os.walk('./FullDataset/Annotations/'):
    paths = [Path(p)/f for f in fs if (Path(p)/f).suffix == '.xml']
    xml_list.extend(paths)

def fix_xml(path:Path,xml_tree:ET.ElementTree):
    # rel_path = path.relative_to('./FullDataset/Annotations/')
    # rel_path = rel_path.with_suffix('.PNG')
    # xml_tree.find('filename').text = str(rel_path)
    objects = xml_tree.findall('object')
    for obj in objects:
        if obj.find('name').text == 'Pliers':
            obj.find('name').text = 'Plier'
        if obj.find('name').text == 'Adjustable Clamp':
            obj.find('name').text = 'AdjustableClamp'
        label = obj.find('name').text
        labels[ label ] += 1
    return labels


labels = defaultdict(int)
for path in xml_list:
    with open(path,'r', encoding='utf-8') as f:
        xml = f.read()
    xml_tree = ET.parse(path)
    fix_xml(path,xml_tree)
    xml_tree.write(path)

print(labels)
