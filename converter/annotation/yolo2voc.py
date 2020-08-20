#  Script to convert yolo annotations to voc format
import os
import xml.etree.cElementTree as ET
from PIL import Image
import numpy as np
from util.filecontrol import *
from typing import *


class InputFileFormatError(Exception):
    def __str__(self):
        return 'inputpath가 존재하지 않는 경로이거나 폴더나 텍스트파일이 아닙니다.'

def loadlabel(inputpath):
    if os.path.isdir(inputpath):
        dirpath = inputpath
    elif os.path.isfile(inputpath) & inputpath.endswith('.txt'):
        dirpath = os.path.splitext(inputpath)[0]
    else:
        raise InputFileFormatError
    with open(classfile,'r') as f:
        labels = f.readlines()
    labels = [i.rstrip('\n') for i in labels]
    ids = [str(i) for i in list(range(len(labels)))]
    labeldict = dict(zip(ids, labels))

    return labeldict

def calLTRB(yolobbox) -> np.ndarray:
    #calculate Left, Top, Right, Bottom
    x, y, w, h = yolobbox
    xExtent = w/2.
    yExtent = h/2.
    L = x-xExtent
    R = x+xExtent
    T = y-yExtent
    B = y+yExtent
    LTRB = np.array([L, T, R, B])
    return LTRB

def create_root(imgpath, width, height):
    imgname = os.path.basename(imgpath)
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = imgname
    ET.SubElement(root, "folder").text = None
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(imgpath, width, height, voc_labels):
    root = create_root(imgpath, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    return tree

def read_file(imgpath):
    path, imgname = os.path.split(imgpath)
    filename = os.path.splitext(imgname)[0]

    img = Image.open(imgpath)

    w, h = img.size
    prueba = f"{path}/{filename}.txt"
    with open(prueba) as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:	
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            LTRB = calLTRB(data[1:])
            voc.extend(LTRB)
            voc_labels.append(voc)
        tree = create_file(imgpath, w, h, voc_labels)
        path, imgname = os.path.split(imgpath)
        filename = os.path.splitext(imgname)[0]
        tree.write("{}/{}.xml".format(path, filename))
    print("Processing complete for file: {}".format(imgpath))


def start(textfile,root):
    imgpathlist = txt2list(textfile)
    imgpathlist = map(lambda path: f'{root}/{path}', imgpathlist)

    for imgpath in imgpathlist:
        path, _ = os.path.split(imgpath)
        if not os.path.exists(path):
            os.makedirs(path)
        read_file(imgpath) 


if __name__ == "__main__":
    textfile = "/home/tm/Code/darknet/data/test.txt"
    root = '/home/tm/Code/darknet'
    classfile = "/home/tm/Code/darknet/data/obj.names"
    CLASS_MAPPING = loadlabel(textfile)
    start(textfile, root)