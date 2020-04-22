
import pandas as pd
import numpy as np
from util import *
from sklearn.metrics import average_precision_score
from os.path import isfile

# def main(pred, img, img_size, runtime, GT=None, IOU_thrsh=0.5, conf_thrsh=0.25):
#     perObj_table = 
#     perImg_table = 
#     perDts_table = 

# def mk_perObj_table(pred, img, img_size, runtime, GT=None, IOU_thrsh=0.5, conf_thrsh=0.25):
#     pred_part = mk_pred
#     gt_part = 
#     eval_part = 

def get_pred(pred, img, img_size, runtime, model='yolo'):
    assert model in ['yolo'], 'Check using model'
    '''
    yolo pred: [('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)),
                ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px)),
                .....]
    '''
    if model=='yolo':
        pred = pd.DataFrame(pred, columns=['class', 'conf', 'bbox'])
        bbox = pd.DataFrame(list(pred['bbox'].values), 
                            columns=['pred_x_center', 'pred_y_center', 'pred_W', 'pred_H'])
        pred.drop(columns='bbox',inplace=True)
        pred = pd.concat([pred, bbox], axis=1)

    pred['img'] = img
    pred['img_W'] = img_size[0]
    pred['img_H'] = img_size[1]
    pred['runtime'] = runtime
    return pred

def calLTRB(yolobbox):
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

def process_pred(pred):
    # pred = pred.reset_index().rename(columns={'index':'pred_id'})
    pred = pred.reset_index(drop=True)
    xyWH_cols = ['pred_x_center', 'pred_y_center', 'pred_W', 'pred_H']

    LTRB = pred[xyWH_cols].apply(calLTRB, axis=1)
    LTRB_table = pd.DataFrame(list(LTRB.values),columns=['pred_left','pred_top','pred_right','pred_bottom']).apply(np.around).astype('int')

    pred[xyWH_cols] = pred[xyWH_cols] / pred[['img_W','img_H']*2].values
    
    pred_part = pd.concat([pred, LTRB_table], axis=1)
    return pred_part

def get_gt(perObj_predpart):
    xyWH_cols = ['gt_x_center', 'gt_y_center', 'gt_W', 'gt_H']
    gt = pd.DataFrame()
    for img in perObj_predpart['img'].unique():
        anno = chgext(img, '.txt')
        if isfile(anno):
            gtInimg = map(lambda gt: gt.split(' '),txt2list(anno))
            gt_sub = pd.DataFrame(gtInimg, columns=['class', *xyWH_cols])
            gt_sub['class'] = gt_sub['class'].astype('int')
            gt_sub[xyWH_cols] = gt_sub[xyWH_cols].astype('float')
            gt_sub['img'] = img
            gt_sub['img_W'] = perObj_predpart.loc[perObj_predpart['img']==img, 'img_W'].iloc[0]
            gt_sub['img_H'] = perObj_predpart.loc[perObj_predpart['img']==img, 'img_H'].iloc[0]
            gt = pd.concat([gt, gt_sub], ignore_index=False)
    if gt.count().sum()==0:
        return None
    else:
        return gt

def process_gt(perObj_predpart, namesfile):
    gt = get_gt(perObj_predpart)
    gt = gt.reset_index().rename(columns={'index':'id'})
    classes = txt2list(namesfile)

    gt['class'] = gt['class'].map(lambda idx: classes[idx])

    ltrb_cols = ['gt_left','gt_top','gt_right','gt_bottom']
    xyWH_cols = ['gt_x_center', 'gt_y_center', 'gt_W', 'gt_H']
    
    LTRB = gt[xyWH_cols].apply(calLTRB, axis=1)
    LTRB_table = pd.DataFrame(list(LTRB.values),columns=ltrb_cols)
    gt_part = pd.concat([gt, LTRB_table], axis=1)

    # print(gt[ltrb_cols])
    # print(gt[['img_W','img_H']*2].values)
    gt_part[ltrb_cols] = (gt_part[ltrb_cols] * gt_part[['img_W','img_H']*2].values).apply(np.around).astype('int')
    return gt_part

def IOU(Bbox1, Bbox2):
    '''
    Bbox = (left, top, right, bottom)
    '''
    gtWrange = set(range(Bbox1[0], Bbox1[2]+1))
    gtHrange = set(range(Bbox1[1], Bbox1[3]+1))
    predWrange = set(range(Bbox2[0], Bbox2[2]+1))
    predHrange = set(range(Bbox2[1], Bbox2[3]+1))

    Wintersection = gtWrange & predWrange
    Hintersection = gtHrange & predHrange

    Area1 = (Bbox1[2]+1-Bbox1[0])*(Bbox1[3]+1-Bbox1[1])
    Area2 = (Bbox2[2]+1-Bbox2[0])*(Bbox2[3]+1-Bbox2[1])
    intersectionArea = len(Wintersection)*len(Hintersection)

    iou = intersectionArea / (Area1+Area2-intersectionArea)
    return iou

def eval(pred_part, namesfile, IOU_thrsh=0.5):
    gt_part = process_gt(pred_part, namesfile)
    matching_result = matching(pred_part, gt_part, IOU_thrsh)
    eval_condition(matching_result)
    matching_result.to_csv('perObj.csv', encoding='euc-kr')
    return matching_result


def matching(pred_part, gt_part, IOU_thrsh):
    pred_ltrb = ['pred_left','pred_top','pred_right','pred_bottom']
    gt_ltrb = ['gt_left','gt_top','gt_right','gt_bottom']
    pred_part['IoU'] = np.nan

    for img in pred_part['img'].unique():
        for cat in pred_part['class'].unique():
            row_condition = (pred_part['img']==img) & (pred_part['class']==cat)
            pred = pred_part.loc[row_condition]
            for index,value in gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cat)].T.items():
                GTBbox = value[gt_ltrb]
                iou = pred[pred_ltrb].apply(lambda predBbox: IOU(GTBbox, predBbox),axis=1)
                # iou = pd.Series(iou, index=pred['pred_id'])
                if not iou.count == 0:
                    iouidx = iou.idxmax()
                    ioumax = iou.max()
                    if ((IOU_thrsh <= ioumax) & (ioumax> pred.fillna(-1).loc[iouidx, 'IoU'].item())):
                        pred_part.loc[iouidx, 'id'] = gt_part.loc[index,'id']
                        pred_part.loc[iouidx, 'IoU'] = ioumax

    matching_result = pd.merge(pred_part, gt_part, how='outer')
    return matching_result

def eval_condition(matching_result):
    matching_result.loc[~matching_result['IoU'].isna(),'eval'] = 'TP'
    matching_result.loc[matching_result['id'].isna() & matching_result['IoU'].isna(),'eval'] = 'FP'
    matching_result.loc[~matching_result['id'].isna() & matching_result['IoU'].isna(),'eval'] = 'FN'

def to_perImg(perObj, conf_thrsh=0.25):
    perImg = pd.DataFrame()
    perObj = perObj.loc[perObj['conf']>=conf_thresh]
    common_cols = ['img', 'img_W','img_H', 'runtime']
    avg_cols = ['conf', 'gt_W', 'gt_H', 'IoU']
    min_cols = ['gt_W', 'gt_H']
    max_cols = ['gt_W', 'gt_H']
    count_cols = ['class', 'eval']

        out = perObject.loc[perObject['img']==img, base].value_counts()
        if prefix:
            out = out.add_prefix(f'{prefix}_')
        return out

    for img in perObj['image'].unique():
        perImg['class'] = perObj['class'].unique().append('total')
        pred_count = perObj.loc[(perObj['img']==img) & (perObj['eval'].map(lambda eval: eval in ['TP', 'FP'])), 'class'].value_counts()
        perImg['pred_count'] = perImg['class'].map(lambda class: pred_count[class])
        gt_count = perObj.loc[(perObj['img']==img) & (perObj['eval'].map(lambda eval: eval in ['TP', 'FN'])), 'class'].value_counts()
        perImg['gt_count'] = perImg['class'].map(lambda class: gt_count[class])
        gt_count = p
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