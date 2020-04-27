import pandas as pd
import numpy as np
from util import *
from sklearn.metrics import average_precision_score
from os.path import isfile
import time
import datetime as dt

# def main(pred, img, img_size, runtime, GT=None, IOU_thresh=0.5, conf_thresh=0.25):
#     perObj_table = 
#     perImg_table = 
#     perDts_table = 

# def mk_perObj_table(pred, img, img_size, runtime, GT=None, IOU_thresh=0.5, conf_thresh=0.25):
#     pred_part = mk_pred
#     gt_part = 
#     eval_part = 

save_csv = lambda df, path: df.to_csv(path, encoding='euc-kr', index=False)
load_csv = lambda path: pd.read_csv(path, encoding='euc-kr')

def run_detection(function, imgtxtpath, detection_time):
    imgpath = txt2list(imgtxtpath)
    subset = pickFilename(imgtxtpath)
    pred = pd.DataFrame()
    starttime = time.time()
    pred = pd.DataFrame()
    for n, i in enumerate(imgpath):
        print(n,'\t', i)
        result, img_W, img_H = function(i)
        runtime = time.time()-starttime
        print("pred time : {}".format(runtime))
        pred_sub = get_pred(result, i, (img_W, img_H), runtime, model='yolo')
        pred = pred.append(pred_sub)
        starttime = time.time()
    save_csv(pred, f'{detection_time}_{subset}_pred.csv')
    return pred

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
    if isinstance(pred,str):
        if isfile(pred):
            pred = pd.read_csv(pred)
        else:
            print('check predict file path')
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
            gt = gt.append(gt_sub, ignore_index=False)
            # gt = pd.concat([gt, gt_sub], ignore_index=False)
    if gt.count().sum()==0:
        return None
    else:
        return gt

def process_gt(perObj_predpart, namesfile, subset):
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
    save_csv(gt_part, f'gtpart_{subset}.cvs')
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

def eval(pred_part, namesfile, imgtxtpath, detection_time, IOU_thresh=0.5, conf_thresh=0.25, load_gt_part=None):
    subset = pickFilename(imgtxtpath)
    if load_gt_part==None:
        gt_part = process_gt(pred_part, namesfile,subset)
    else:
        gt_part = pd.read_csv(load_gt_part)
    matching_result = matching(pred_part, gt_part, IOU_thresh, conf_thresh)
    eval_condition(matching_result)
    subset = pickFilename(subset)
    save_csv(matching_result, f'{detection_time}_{subset}_perObj_It{IOU_thresh}_ct{conf_thresh}.csv')
    return matching_result


def matching(pred_part, gt_part, IOU_thresh, conf_thresh):
    pred_ltrb = ['pred_left','pred_top','pred_right','pred_bottom']
    gt_ltrb = ['gt_left','gt_top','gt_right','gt_bottom']
    pred_part = pred_part.loc[pred_part['conf']>=conf_thresh]
    pred_part.loc[:,'IoU'] = np.nan
    pred_part.loc[:,'id'] = np.nan
    

    for img in pred_part.loc[:,'img'].unique():
        for cat in pred_part.loc[:,'class'].unique():
            row_condition = (pred_part['img']==img) & (pred_part['class']==cat)
            pred = pred_part.loc[row_condition]
            for index,value in gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cat)].iterrows():
                GTBbox = value[gt_ltrb]
                iou = pred[pred_ltrb].apply(lambda predBbox: IOU(GTBbox, predBbox),axis=1)
                # iou = pd.Series(iou, index=pred['pred_id'])
                if not iou.count == 0:
                    iouidx = iou.idxmax()
                    ioumax = iou.max()
                    if ((IOU_thresh <= ioumax) & (ioumax > pred.fillna(-1).loc[iouidx, 'IoU'].item())):
                        pred_part.loc[iouidx, 'id'] = gt_part.loc[index,'id']
                        pred_part.loc[iouidx, 'IoU'] = ioumax

    matching_result = pd.merge(pred_part, gt_part, how='outer')
    return matching_result

def eval_condition(matching_result):
    matching_result.loc[matching_result['IoU'].notna(),'eval'] = 'TP'
    matching_result.loc[matching_result['id'].isna() & matching_result['IoU'].isna(),'eval'] = 'FP'
    matching_result.loc[matching_result['id'].notna() & matching_result['IoU'].isna(),'eval'] = 'FN'

def to_other(perObj, imgtxtpath, detection_time, IOU_thresh, conf_thresh, other=''):
    assert other in ['perImg', 'perDataset'], f'check argument \'other\'. other can\'t be {other}'
    subset = pickFilename(imgtxtpath)

    gt_rows = perObj.loc[perObj['id'].notna()]
    pred_rows = perObj.loc[perObj['conf'].notna()]

    groupby_cols = {'perImg_class': ['img','class'], 
                    'perImg_total':['img'],
                    'perDataset_class': ['class'],
                    'perDataset_total': []}

    def do_groupby(df, by_col, total=False):
        # if other=='perImg':
        #     gr


        # if total:
        #     gt_count = gt_rows.groupby('img')['class'].count().rename('gt_count')
        #     pred_count = pred_rows.groupby('img')['class'].count().rename('pred_count')
        # else:
        #     gt_count = gt_rows.groupby('img')['class'].value_counts().rename('gt_count')
        #     pred_count = pred_rows.groupby('img')['class'].value_counts().rename('pred_count')

        perObj_groupby = df.groupby(by_col)
        gt_rows['gt_count'] = 1
        pred_rows['pred_count'] = 1
        gt_count = gt_rows.groupby(by_col).count()['gt_count']
        pred_count = pred_rows.groupby(by_col).count()['pred_count']
        eval_count = perObj_groupby['eval'].value_counts().unstack().add_suffix('_count').fillna(0)
        
        minmaxavg_cols = ['conf', 'pred_W', 'pred_H', 'IoU','gt_W', 'gt_H']
        minmaxavg_groups = perObj_groupby[minmaxavg_cols]
        min_table = minmaxavg_groups.min().add_prefix('min_')
        avg_table = minmaxavg_groups.mean().add_prefix('avg_')
        max_table = minmaxavg_groups.max().add_prefix('max_')

        recall = (eval_count['TP_count'] / (eval_count['TP_count']+eval_count['FN_count'])).rename('recall')
        precision = (eval_count['TP_count'] / (eval_count['TP_count']+eval_count['FP_count'])).rename('precision')
        F1 = (2*recall*precision / (recall+precision)).rename('F1-Score')
        # ap = 

        output = pd.concat([gt_count, pred_count, eval_count, recall, precision, F1, 
                            avg_table, avg_table, avg_table], axis=1)
        if total:
            output.index = pd.MultiIndex.from_product([output.index.tolist(), ['total']], names=['img', 'class'])
        output.reset_index(inplace=True)
        return output

    per_class = do_groupby(perObj, groupby_cols[f'{other}_class'], total=False)
    total = do_groupby(perObj, groupby_cols[f'{other}_total'], total=True)
    other_table = per_class.append(total).sort_values(['img', 'class'], ignore_index=True)

    other_table = pd.merge(other_table, perObj[['img','img_W', 'img_H', 'runtime']].drop_duplicates('img'), how='left')
    save_csv(other_table, f'{detection_time}_{subset}_{other}_It{IOU_thresh}_ct{conf_thresh}.csv')