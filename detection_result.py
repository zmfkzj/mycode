import pandas as pd
import numpy as np
from util import *
from sklearn.metrics import average_precision_score

perObject_columns = ['image', 'img_H','img_W', 
                    'id','GT_class','GT_x_center', 'GT_y_center', 'GT_W', 'GT_H',
                    'GT_top','GT_left','GT_bottom','GT_right',
                    'Infer_class','Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H','conf',
                    'Infer_left','Infer_top','Infer_right','Infer_bottom',
                    'IoU', 'TP/FP/FN']
perImage_columns = ['image', 'img_H','img_W','runtime',
                    'GT_class','GT_count'
                    'Infer_class','Infer_count', 'avg_conf',
                    'avg_IoU', 'sum_TP','sum_FP','sum_FN']
perDataset_columns = ['runtime', 'img_count',
                    'GT_class','GT_count'
                    'Infer_class','Infer_count', 'avg_conf',
                    'avg_IoU', 'sum_TP','sum_FP','sum_FN']
perClass_columns = ['GT_class']

detectionResult_perObject = pd.DataFrame(columns=perObject_columns)
detectionResult_perImage = pd.DataFrame(columns=perImage_columns)
detectionResult_perDataset = pd.DataFrame(columns=perDataset_columns)



def today():
    detectionResult_perObject['infer_date']=pd.datetime.today().strftime('%y%m%d-%H-%M')
    detectionResult_perImage['infer_date']=pd.datetime.today().strftime('%y%m%d-%H-%M')
    detectionResult_perDataset['infer_date']=pd.datetime.today().strftime('%y%m%d-%H-%M')

def yoloOutput(output, img_W, img_H):
    '''
    yolo output: [('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)),
                ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)),
                .....]
    '''
    Class, conf, bbox = list(zip(*output))
    x, y, w, h = list(zip(*bbox))
    columns = ['Infer_class', 'conf', 'Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']
    # L, T, R, B = list(zip(*list(map(map(int,calLTRB),bbox))))
    data = [Class, conf, x, y, w, h]
    cvtOutput = pd.DataFrame(dict(zip(columns, data)))
    # cvtOutput[['Infer_left','Infer_top','Infer_right','Infer_bottom']] = cvtOutput[['Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']].apply(calLTRB,axis=1)
    LTRB = list(zip(*cvtOutput[['Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']].apply(calLTRB, axis=1).map(lambda x: np.around(x).astype('int32'))))
    LTRBdf = pd.DataFrame(dict(zip(['Infer_left','Infer_top','Infer_right','Infer_bottom'], LTRB)))
    cvtOutput = pd.concat([cvtOutput, LTRBdf],axis=1)
    cvtOutput[['Infer_x_center', 'Infer_W']] /= img_W
    cvtOutput[['Infer_y_center', 'Infer_H']] /= img_H
    cvtOutput['id'] = np.nan
    return cvtOutput

def mkInfer_subdf(output, img_W, img_H, model=None):
    '''
    output columns = ['Infer_class', 'conf', 'Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H',
                    'Infer_left','Infer_top','Infer_right','Infer_bottom',
                    'img_W', 'img_H', 'id']

    var output form = [(Class1, cofidence1, left1, top1, right1, bottom1), 
                        (Class2, cofidence2, left2, top2, right2, bottom2)]
    '''
    assert model in ['yolo'], 'check model'
    if model=='yolo':
        output = yoloOutput(output, img_W, img_H)

    Class, conf, bbox = list(zip(*output))
    x, y, w, h = list(zip(*bbox))
    columns = ['Infer_class', 'conf', 'Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']
    # L, T, R, B = list(zip(*list(map(map(int,calLTRB),bbox))))
    data = [Class, conf, x, y, w, h]
    cvtOutput = pd.DataFrame(dict(zip(columns, data)))
    # cvtOutput[['Infer_left','Infer_top','Infer_right','Infer_bottom']] = cvtOutput[['Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']].apply(calLTRB,axis=1)
    LTRB = list(zip(*cvtOutput[['Infer_x_center', 'Infer_y_center', 'Infer_W', 'Infer_H']].apply(calLTRB, axis=1).map(lambda x: np.around(x).astype('int32'))))
    LTRBdf = pd.DataFrame(dict(zip(['Infer_left','Infer_top','Infer_right','Infer_bottom'], LTRB)))
    cvtOutput = pd.concat([cvtOutput, LTRBdf],axis=1)
    cvtOutput[['Infer_x_center', 'Infer_W']] /= img_W
    cvtOutput[['Infer_y_center', 'Infer_H']] /= img_H
    cvtOutput['id'] = np.nan
    return cvtOutput

def calLTRB(originBbox):
    #calculate Left, Top, Right, Bottom
    yExtent = originBbox[3]
    xExtent = originBbox[2]
    # Coordinates are around the center
    xCoord = originBbox[0] - originBbox[2]/2
    if xCoord<0: xCoord = 0
    yCoord = originBbox[1] - originBbox[3]/2
    if yCoord<0: yCoord = 0
    LTRB = np.array([xCoord, yCoord, xCoord + xExtent, yCoord + yExtent])
    return LTRB

def mkSubGTdf(anno, namesfile, img_W, img_H):
    GTlist = list(map(lambda str: list(map(num, str.split(' '))), txt2list(anno)))
    classes = txt2list(namesfile)
    for gt in GTlist:
        gt[0] = classes[int(gt[0])]
    GT_columns = ['GT_class','GT_x_center', 'GT_y_center', 'GT_W', 'GT_H']
    GTdict = dict(zip(GT_columns, list(zip(*GTlist))))
    subGTdf = pd.DataFrame(GTdict)
    LTRB = list(zip(*subGTdf[['GT_x_center', 'GT_y_center', 'GT_W', 'GT_H']].apply(lambda originBbox: calLTRB(originBbox)*np.array([img_W, img_H]*2), axis=1).map(lambda x: np.around(x).astype('int32'))))
    LTRBdf = pd.DataFrame(dict(zip(['GT_left','GT_top','GT_right','GT_bottom'], LTRB)))
    subGTdf = pd.concat([subGTdf, LTRBdf],axis=1)

    return subGTdf

def iou(GTBbox, InferBbox):
    '''
    Bbox = (left, top, right, bottom)
    '''
    gtWrange = set(range(GTBbox[0], GTBbox[2]+1))
    gtHrange = set(range(GTBbox[1], GTBbox[3]+1))
    InferWrange = set(range(InferBbox[0], InferBbox[2]+1))
    InferHrange = set(range(InferBbox[1], InferBbox[3]+1))

    Wintersection = gtWrange & InferWrange
    Hintersection = gtHrange & InferHrange

    gtArea = (GTBbox[2]+1-GTBbox[0])*(GTBbox[3]+1-GTBbox[1])
    InferArea = (InferBbox[2]+1-InferBbox[0])*(InferBbox[3]+1-InferBbox[1])
    intersectionArea = len(Wintersection)*len(Hintersection)

    IOU =intersectionArea/(gtArea+InferArea-intersectionArea) * 100

    return IOU

def TPFPFN(result_perObject):
    result_perObject.loc[result_perObject['GT_class']==result_perObject['Infer_class'],'TP/FP/FN'] = 'TP'
    result_perObject.loc[result_perObject['GT_class'].isna(),'TP/FP/FN'] = 'FP'
    result_perObject.loc[result_perObject['Infer_class'].isna(),'TP/FP/FN'] = 'FN'

def to_perImg(perObject_allconf, conf_thresh=0.25):
    perObject = perObject_allconf.loc[perObject_allconf['conf']>=conf_thresh]
    Slice = perObject[['image', 'img_H', 'img_W', 'GT_class', 'Infer_class', 'conf', 'IoU','TP/FP/FN']]
    def count(base, prefix=None):
        out = perObject.loc[perObject['image']==img, base].value_counts().to_frame(name=img).T
        if prefix:
            out = out.add_prefix(f'{prefix}_')
        return out

    subImgs = []
    for img in perObject['image'].unique():
        GT_count = count('GT_class', 'GT')
        Infer_count = count('Infer_class', 'Infer')
        TPFPFN_count = count('TP/FP/FN')
        subImg = pd.concat([GT_count, Infer_count, TPFPFN_count], axis=1)
        subImgs.append(subImg)

    result_perImg = pd.concat(subImgs).fillna(0) 
    result_perImg['Precision'] = result_perImg['TP']/(result_perImg['TP']+result_perImg['FP'])
    result_perImg['Recall'] = result_perImg['TP']/(result_perImg['TP']+result_perImg['FN'])
    result_perImg['F1-score'] = 2*result_perImg['Precision']*result_perImg['Recall']/(result_perImg['Precision']+result_perImg['Recall'])
    result_perImg['']

    print('\n', result_perImg)


