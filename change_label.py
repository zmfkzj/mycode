import os
import xml.etree.ElementTree as ET


change_list = {'MS(Material Seperation)' : 'MS',
                'RE(Rebar Exposure)' : 'RE'}

def analyze_anno(imgpath_textfile):
    with open(imgpath_textfile, 'r') as f:
        xmlpathlist = f.readlines()
    xmlpathlist = [path.rstrip('\n') for path in xmlpathlist]
    xmlpathlist = [f'{os.path.splitext(path)[0]}.xml' for path in xmlpathlist]

    classes = dict()
    for xmlfile in xmlpathlist:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        for obj in root.iter('object'):
            label = obj.find('name').text
            if label in change_list.keys():
                obj.find('name').text = change_list[label]
                label = change_list[label]
            if label not in classes.keys():
                classes[label] = 1
            else:
                classes[label] += 1
    for key, value in classes.items():
        print(f'{key}\tObject가 총 {value}개 있습니다.')

def change_label(path, before_label, after_label):
    xml = ET.parse(path)
    root = xml.getroot()
    for obj in root.iter('object'):
        label = obj.find('name').text
        if label in change_list.keys():
            obj.find('name').text = change_list[label]
    xml.write(path)

if __name__ == "__main__":
    analyze_anno('/home/tm/dataset/Crack/최종발표대비/panorama_all.txt')
    # change_label('/run/user/1000/gvfs/ftp:host=nas62.local/File/결함-YOLOv3/stillcut/191023_당인교_윈치캠영상/C0005 (12-10-2019 9-15-04 AM)_all.txt')
    # change_label('/run/user/1000/gvfs/ftp:host=nas62.local/File/결함-YOLOv3/stillcut/191023_당인교_윈치캠영상/C0005 (12-10-2019 9-15-04 AM)/C0005 034.xml',
            # list(change_list.keys())[0], list(change_list.values())[0])