import json
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))

from util.filecontrol import folder2list, chgext
from collections import defaultdict
from converter.annotation.yolo2voc import create_file
import cv2

'''
SQ로부터 받은 json 파일을 voc format의 xml 파일로 변환
'''
#-----------------------------------------
root = osp.expanduser('~/host/nasrw/터널결함정보/SQ수령 이미지(20200805)-5차/1차(BBOX, SEGMENT) 샘플/Defects')
jsonFiles = folder2list(root, extlist=['.json'])
#-----------------------------------------

def loadJson(path):
    with open(path) as f:
        data = json.load(f)
    return data

if __name__ == "__main__":

    for jsonFile in jsonFiles:
        jsonFileFullPath = osp.join(root, jsonFile)
        category, filename = osp.split(jsonFile)
        data = loadJson(jsonFileFullPath)
        bboxes = []
        for el in data:
            el = defaultdict(bool, el)
            if el['type']=='box':
                el['array'][0] = category
                el['array'][3] = el['array'][1] + el['array'][3]
                el['array'][4] = el['array'][2] + el['array'][4]
                bboxes.append(el['array'])
        if bboxes:
            imgPath = chgext(jsonFileFullPath, '.jpg')
            img = cv2.imread(imgPath)
            h, w, _ = img.shape
            tree = create_file(imgPath, w, h, bboxes)
            tree.write(chgext(jsonFileFullPath, '.xml'))
