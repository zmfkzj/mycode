import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from glob import glob
from util.filecontrol import txt2list, chgext
import numpy as np

# classes = ["Crack", "Leakage" ,"Peeling", "Desqu", "Efflor", "Fail", "MS", "RE"]

def calxyWH(ltbr_bbox, img_size):
    '''
    ltbr_bbox = (left, top, right, bottom)
    img_size = (H, W)
    '''
    l, t, r, b = ltbr_bbox
    dh = 1./img_size[0]
    dw = 1./img_size[1]
    x = (l + r)/2.0
    y = (t + b)/2.0
    w = r - l
    h = b - t
    xywh = np.array([x*dw, y*dh, w*dw, h*dh])
    return xywh

def load_annotation(xmlfile:str) -> dict:
    in_file = open(xmlfile)
    tree=ET.parse(in_file)
    root = tree.getroot()
    anno = dict()
    size = root.find('size')
    anno['width'] = int(size.find('width').text)
    anno['height'] = int(size.find('height').text)

    objs = []
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        obj = (cls, float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        objs.append(obj)

    anno['obj'] = objs
    return anno

def convert_annotation(xmlfile, classes):
    anno = load_annotation(xmlfile)
    objs = anno['obj']
    h = anno['height']
    w = anno['width']
    out_file = open(xmlfile[:-3]+'txt', 'w')
    for obj in objs:
        bb = calxyWH(obj[1:] , (h,w))
        out_file.write(obj[0] + " " + " ".join([str(a) for a in bb]) + '\n')

def main(imgtxtlist, namesfile):
    root = os.path.commonpath
    xmlpathlist = txt2list(imgtxtlist)
    xmlpathlist = chgext(xmlpathlist, '.xml')

    for root in xmlpathlist:
        convert_annotation(root)

if __name__ == "__main__":
    # main()
    pass
