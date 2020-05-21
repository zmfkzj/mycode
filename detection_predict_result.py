import pandas as pd
import numpy as np
from util.filecontrol import *
from converter.annotation.voc2yolo import calxyWH
from converter.annotation.yolo2voc import calLTRB
from sklearn.metrics import average_precision_score
from os.path import isfile, join, basename
import time
import datetime as dt
from itertools import product, chain
from collections import defaultdict
import datetime as dt
import xml.etree.ElementTree as ET

load_csv = lambda path: pd.read_csv(path, encoding='euc-kr')

def save_csv_yolo(df, path, root, **kwargs):
    df.to_csv(join(root, 'data', path), encoding='euc-kr', index=False)

def save_csv_voc(df, path, root, **kwargs):
    df.to_csv(join(root, path), encoding='euc-kr', index=False)

def run_detection(function, imgtxtpath, detection_time, configPath, weightPath, root='', **kwargs):
    pathlist = txt2list(imgtxtpath)
    subset = pickFilename(imgtxtpath)
    pred = pd.DataFrame()
    starttime = time.time()
    pred = pd.DataFrame()
    img_count = len(pathlist)
    for idx, imgpath in enumerate(pathlist):
        path = join(root, imgpath)
        print(f'{idx}/{img_count}','\t', path)
        result, img_W, img_H = function(path)
        runtime = time.time()-starttime
        starttime = time.time()
        print("pred time : {}".format(runtime))
        pred_sub = get_pred(result, path, (img_W, img_H), runtime, model='yolo')
        pred = pred.append(pred_sub)
    weight = pickFilename(weightPath).split('_')[-1]
    save_csv_yolo(pred, f'{subset}_pred_{pickFilename(configPath)}_{weight}_{detection_time}.csv', root=root)
    return pred

def get_pred(pred, img_path, img_size, runtime, model='yolo'):
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

    pred['img'] = img_path
    pred['img_W'] = img_size[0]
    pred['img_H'] = img_size[1]
    pred['runtime'] = runtime
    return pred

def process_pred(pred):
    if isinstance(pred,str):
        if isfile(pred):
            pred = load_csv(pred)
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

def loadbbox(annofile, form='txt'):
    if form=='txt':
        gtInimg = map(lambda gt: gt.split(' '),txt2list(annofile))
    elif form=='xml':
        tree = ET.parse(annofile)
        xmlroot = tree.getroot()
        size = xmlroot.find('size')
        img_W = int(size.find('width').text)
        img_H = int(size.find('height').text)

        gtInimg = []
        for obj in xmlroot.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = (cls, float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text), img_W, img_H)
            gtInimg.append(b)
    return gtInimg

def get_gt(file_list, form, root, **kwargs):
    '''
    file_list is predpart:DataFrame or list of img path(at yolo), filename(at voc).
    root is data/..(yolo), (ImagesSet, Annotations, ...)/..(voc)
    '''

    gt = pd.DataFrame()
    if isinstance(file_list, type(pd.DataFrame())):
        files = file_list['img'].unique()
    elif isinstance(file_list, str):
        files = txt2list(file_list)
    elif isinstance(file_list, list):
        files = file_list

    xyWH_cols = ['gt_x_center', 'gt_y_center', 'gt_W', 'gt_H']
    LTRB_cols = ['gt_left', 'gt_top', 'gt_right', 'gt_bottom']
    for filename in files:
        if form=='yolo':
            anno = join(root, chgext(filename, '.txt'))

            if isfile(anno):
                gtInimg = loadbbox(anno, 'txt')
                gt_sub = pd.DataFrame(gtInimg, columns=['class', *xyWH_cols])
                gt_sub['class'] = gt_sub['class'].astype('int')
                gt_sub[xyWH_cols] = gt_sub[xyWH_cols].astype('float')
                gt_sub['img'] = filename
                
                if isinstance(file_list, type(pd.DataFrame())):
                    gt_sub['img_W'] = file_list.loc[file_list['img']==filename, 'img_W'].iloc[0]
                    gt_sub['img_H'] = file_list.loc[file_list['img']==filename, 'img_H'].iloc[0]
                elif isinstance(file_list, list):
                    img = cv2.imread(join(root, filename))
                    imgsize = img.shape
                    gt_sub['img_W'] = imgsize[1]
                    gt_sub['img_H'] = imgsize[0]
            else:
                anno = join(root, chgext(filename, '.xml'))
                if isfile(anno):
                    gtInimg = loadbbox(anno, 'xml')
                    gt_sub = pd.DataFrame(gtInimg, columns=['class', *LTRB_cols, 'img_W', 'img_H'])
                    gt_sub[LTRB_cols] = gt_sub[LTRB_cols].astype('int')
                    if isfile(join(root, filename)):
                        gt_sub['img'] = filename

        elif form=='voc':
            anno = join(root, f'Annotations/{filename}.xml')
            if isfile(anno):
                gtInimg = loadbbox(anno, 'xml')
                gt_sub = pd.DataFrame(gtInimg, columns=['class', *LTRB_cols, 'img_W', 'img_H'])
                gt_sub[LTRB_cols] = gt_sub[LTRB_cols].astype('int')
                if isfile(join(root,f'JPEGImages/{filename}.jpg')):
                    gt_sub['img'] = f'JPEGImages/{filename}.jpg'
                elif isfile(join(root,f'JPEGImages/{filename}.png')):
                    gt_sub['img'] = f'JPEGImages/{filename}.png'

        gt = gt.append(gt_sub, ignore_index=False)
    return gt

def process_gt(file_list, subset:str, root, form='yolo', **kwargs):
    assert form in ['yolo', 'voc'], 'form arg is yolo or voc'
    '''
    file_list is predpart:DataFrame or list of img path
    '''
    ltrb_cols = ['gt_left','gt_top','gt_right','gt_bottom']
    xyWH_cols = ['gt_x_center', 'gt_y_center', 'gt_W', 'gt_H']

    gt = get_gt(file_list, form, root)
    gt = gt.reset_index().rename(columns={'index':'id'})

    def process_xml(gt):
        xywh = gt[ltrb_cols+['img_H', 'img_W']].apply(lambda input: calxyWH(input[:4], input[4:]),axis=1)
        xywh_table = pd.DataFrame(list(xywh.values),columns=xyWH_cols)
        gt_part = pd.concat([gt, xywh_table], axis=1)
        return gt_part

    def process_txt(gt):
        namesfile = kwargs['namesfile']
        classes = txt2list(namesfile)
        gt['class'] = gt['class'].map(lambda idx: classes[idx])

        LTRB = gt[xyWH_cols].apply(calLTRB, axis=1)
        LTRB_table = pd.DataFrame(list(LTRB.values),columns=ltrb_cols)
        gt_part = pd.concat([gt, LTRB_table], axis=1)
        gt_part[ltrb_cols] = (gt_part[ltrb_cols] * gt_part[['img_W','img_H']*2].values).apply(np.around).astype('int')
        return gt_part

    if form=='yolo':
        if gt['class'].dtype=='int':
            gt_part = process_txt(gt)
        else:
            gt_part = process_xml(gt)
        save_csv_yolo(gt_part, f'gtpart_{subset}.csv', root)

    elif form=='voc':
        gt_part = process_xml(gt)
        save_csv_voc(gt_part, f'gtpart_{subset}.csv', root)



    return gt_part

def IOU(Bbox1, Bbox2):
    '''
    Bbox = (left, top, right, bottom)
    '''
    gtW_range = set(range(Bbox1[0], Bbox1[2]+1))
    gtH_range = set(range(Bbox1[1], Bbox1[3]+1))
    predW_range = set(range(Bbox2[0], Bbox2[2]+1))
    predH_range = set(range(Bbox2[1], Bbox2[3]+1))

    W_intersection = gtW_range & predW_range
    H_intersection = gtH_range & predH_range

    Area1 = (Bbox1[2]+1-Bbox1[0])*(Bbox1[3]+1-Bbox1[1])
    Area2 = (Bbox2[2]+1-Bbox2[0])*(Bbox2[3]+1-Bbox2[1])
    intersectionArea = len(W_intersection)*len(H_intersection)

    iou = intersectionArea / (Area1+Area2-intersectionArea)
    return iou

def to_perObj(pred_part, imgtxtpath, detection_time, IOU_thresh=0.5, conf_thresh=0.25, load_gt_part=None, root='', **kwargs):
    namesfile = kwargs['namesfile']
    configPath = pickFilename(kwargs['configPath'])
    weight = pickFilename(kwargs['weightPath']).split('_')[-1]

    subset = pickFilename(imgtxtpath)
    if load_gt_part==None:
        gt_part = process_gt(pred_part, subset, root, **kwargs)
    else:
        gt_part = pd.read_csv(load_gt_part)
    matching_result = matching(pred_part, gt_part, IOU_thresh, conf_thresh)
    eval_condition(matching_result)
    save_csv_yolo(matching_result, f'{subset}_perObj_{configPath}_{weight}_It{IOU_thresh}_ct{conf_thresh}_{detection_time}.csv', root)
    return matching_result


def matching(pred_part, gt_part, IOU_thresh, conf_thresh):
    pred_ltrb = ['pred_left','pred_top','pred_right','pred_bottom']
    gt_ltrb = ['gt_left','gt_top','gt_right','gt_bottom']
    pred_part = pred_part.loc[pred_part['conf']>=conf_thresh]
    pred_part['IoU'] = np.nan
    pred_part['id'] = np.nan
    

    for img in pred_part['img'].unique():
        for cat in pred_part['class'].unique():
            row_condition = (pred_part['img']==img) & (pred_part['class']==cat)
            pred = pred_part.loc[row_condition]
            for index,value in gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cat)].iterrows():
                GTBbox = value[gt_ltrb]
                iou = pred[pred_ltrb].apply(lambda predBbox: IOU(GTBbox, predBbox),axis=1)
                # iou = pd.Series(iou, index=pred['pred_id'])
                if not (iou.count() == 0).all():
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

def get_groupby(perObj, cols):
    if cols:
        perObj_groupby = perObj.groupby(cols)
    else:
        perObj_groupby = perObj
    return perObj_groupby

def chg_colname(table):
    table.columns = list(map(lambda idx: '_'.join(idx), table.columns.to_flat_index().values))
    table.rename(columns={'id_count':'gt_count', 'conf_count':'pred_count', 
                        'class_':'class', 'img_':'img'}, inplace=True)

def to_other(perObj, imgtxtpath, detection_time, IOU_thresh, conf_thresh, root, **kwargs):
    subset = pickFilename(imgtxtpath)
    weight = pickFilename(kwargs['weightPath']).split('_')[-1]
    configPath = pickFilename(kwargs['configPath'])

    common_cols = {'perImg':['img','class'],
                    'perDataset':['class']}
    minmaxavg_cols = {'perImg':['conf', 'IoU', 'pred_W', 'pred_H','gt_W', 'gt_H'],
                    'perDataset': ['conf', 'IoU', 'pred_W', 'pred_H','gt_W', 'gt_H']}
    count_cols = {'perImg': ['id', 'conf', 'TP', 'FP', 'FN'],
                'perDataset': ['id', 'conf', 'TP', 'FP', 'FN']}
    paste_cols = {'perImg':['img_W', 'img_H', 'runtime'],
                'perDataset':[]}
    
    numeric_cols = ['min', 'mean','max','std','median']

    evals = pd.get_dummies(perObj['eval']).replace({0:np.nan})
    perObj = pd.concat([perObj.drop(columns='eval'), evals], axis=1)
    perOther = {}
    for other, cols in common_cols.items():
        perObj_groupby_classes = get_groupby(perObj, cols)
        cols.remove('class')
        perObj_groupby_total = get_groupby(perObj, cols)

        count_mcols = dict(product(count_cols[other], [['count']]))
        minmaxavg_mcols = dict(product(minmaxavg_cols[other], [numeric_cols]))
        paste_mcols = dict(product(paste_cols[other], [['max']]))

        agg_dict = defaultdict(list)
        for k, v in chain(count_mcols.items(), minmaxavg_mcols.items(), paste_mcols.items()):
            agg_dict[k].extend(v)
        
        desc_class = perObj_groupby_classes.agg(agg_dict).reset_index()
        desc_total = perObj_groupby_total.agg(agg_dict)

        if cols:
            desc_total = desc_total.reset_index()
        else:
            desc_total = desc_total.unstack().dropna()
        desc_total['class'] = 'total'
        perOther_table = desc_class.append(desc_total,ignore_index=True)
        chg_colname(perOther_table)

        if other=='perDataset':
            img_count_perDataset = perOther['perImg'][['class', 'img']].groupby('class').count()
            runtime_perDataset = perOther['perImg'][['class', 'runtime']].groupby('class').agg(numeric_cols+['sum'])
            chg_colname(runtime_perDataset)
            img_size_perDataset = perOther['perImg'][['class','img_W', 'img_H']].groupby('class').agg(numeric_cols)
            chg_colname(img_size_perDataset)
            etc_perDataset = pd.concat([img_count_perDataset, runtime_perDataset, img_size_perDataset], axis=1).reset_index()
            perOther_table = pd.merge(perOther_table, etc_perDataset)
            perOther_table.rename(columns={'img':'img_count'}, inplace=True)
        elif other=='perImg':
            perOther_table.rename(columns={'runtime_max':'runtime','img_W_max':'img_W','img_H_max':'img_H'}, inplace=True)

        perOther[other] = perOther_table

    for other, table in perOther.items():

        recall = (table['TP_count'] / (table['TP_count']+table['FN_count'])).rename('recall')
        precision = (table['TP_count'] / (table['TP_count']+table['FP_count'])).rename('precision')
        F1 = (2*recall*precision / (recall+precision)).rename('F1-Score')

        table = pd.concat([table, recall, precision, F1], axis=1)
        save_csv_yolo(table, f'{subset}_{other}_{configPath}_{weight}_It{IOU_thresh}_ct{conf_thresh}_{detection_time}.csv', root)

if __name__ == "__main__":
    pred_file = 'data/valid_pred_yolov4_best_200506-1624.csv'
    pred_file_split = basename(pred_file).split('_')
    root = ''
    arg = { 'IOU_thresh' : 0.5,
            'conf_thresh' : 0.1,
            'imgtxtpath' : f'{pred_file_split[0]}.txt',
            'configPath' : f"{pred_file_split[2]}.cfg", 
            'weightPath' : f"{pred_file_split[3]}.weights", 
            'detection_time' : pred_file_split[4],
            'root': root}

    pred = pd.read_csv(pred_file, encoding='euc-kr')
    perObj_predpart = process_pred(pred)
    perObj = to_perObj(perObj_predpart, "data/obj.names", **arg)
    to_other(perObj, **arg)

    get_gt('data/test.txt')