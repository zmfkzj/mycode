import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from glob import glob

classes = ["Crack", "Leakage" ,"Peeling", "Desqu", "Efflor", "Fail", "MS", "RE"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xmlfile):
    in_file = open(xmlfile)
    out_file = open(xmlfile[:-3]+'txt', 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
path = "/home/tm/dataset/Crack/최종발표대비"
with open(f'{path}/classes.txt','w') as f:
    f.write("\n".join(classes))
imgpath_textfile = '/home/tm/dataset/Crack/최종발표대비/panorama/191023_당인교_윈치캠영상_all.txt'
with open(imgpath_textfile, 'r') as f:
    xmlpathlist = f.readlines()

xmlpathlist = [path.rstrip('\n') for path in xmlpathlist]
xmlpathlist = [f'{os.path.splitext(path)[0]}.xml' for path in xmlpathlist]

for path in xmlpathlist:
    convert_annotation(path)

