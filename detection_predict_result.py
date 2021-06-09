import pandas as pd
import numpy as np
from pandas import read_csv
from util.filecontrol import *
from converter.annotation.voc2yolo import calxyWH
from converter.annotation.yolo2voc import calLTRB
from sklearn.metrics import average_precision_score
from os.path import isfile, join, basename
import time
from itertools import product, chain
from collections import defaultdict
import datetime as dt
import xml.etree.ElementTree as ET
from typing import *
from glob import glob
import re
from tqdm import tqdm
from multiprocessing import Pool
import errno

forms = ['yolo', 'voc']

def load_csv(path):
    try:
        df = pd.read_csv(path, encoding='euc-kr')
    except:
        df = pd.read_csv(path, encoding='utf-8')
    return df

def save_csv(df, path, root, **kwargs):
    df.to_csv(join(root, path), encoding='euc-kr', index=False)

def run_detect(function, imgtxtpath, detection_time, configPath, weightPath, root='', **kwargs):
    pathlist = txt2list(imgtxtpath)
    subset = pickFilename(imgtxtpath)
    starttime = time.time()
    pred = pd.DataFrame()
    img_count = len(pathlist)
    for idx, imgpath in enumerate(pathlist):
        print(f'{idx}/{img_count}','\t', imgpath)
        result, img_W, img_H = function(imgpath)
        runtime = time.time()-starttime
        starttime = time.time()
        print("pred time : {}".format(runtime))
        pred_sub = get_pred(result, imgpath, (img_W, img_H), runtime, model='yolo')
        pred = pred.append(pred_sub)
    weight = pickFilename(weightPath).split('_')[-1]
    save_csv(pred, f'{subset}_pred_{pickFilename(configPath)}_{weight}_{detection_time}.csv', root=root)
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

def process_pred(pred_data:Union[str, pd.DataFrame], form='yolo', **kwargs):
    assert form in forms, '데이터셋 양식을 확인하세요.'
    if isinstance(pred_data,str):
        try:
            pred = load_csv(pred_data)
        except FileNotFoundError as e:
            print(e)
            print('파일의 경로를 확인하세요.')
    else:
        pred = pred_data
    pred = pred.reset_index().rename(columns={'index':'pred_id'})
    # pred = pred.reset_index(drop=True)
    xyWH_cols = ['pred_x_center', 'pred_y_center', 'pred_W', 'pred_H']
    LTRB_cols = ['pred_left', 'pred_top', 'pred_right', 'pred_bottom']

    if form=='yolo':
        LTRB = pred[xyWH_cols].apply(calLTRB, axis=1)
        LTRB_table = pd.DataFrame(list(LTRB.values),columns=LTRB_cols)
        pred[xyWH_cols] = pred[xyWH_cols] / pred[['img_W','img_H']*2].values

        pred_part = pd.concat([pred, LTRB_table], axis=1)
    elif form=='voc':
        xyWH = pred[LTRB_cols+['img_W', 'img_H']].apply(lambda row: calxyWH(row[LTRB_cols],row[['img_H', 'img_W']]), axis=1)
        xyWH_table = pd.DataFrame(list(xyWH.values),columns=xyWH_cols)

        pred_part = pd.concat([pred, xyWH_table], axis=1)
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
            anno = os.path.join(root, '..', chgext(filename, '.txt'))

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
                    img = cv2.imread(os.path.join(root, '..', filename))
                    imgsize = img.shape
                    gt_sub['img_W'] = imgsize[1]
                    gt_sub['img_H'] = imgsize[0]
            elif isfile(anno:=os.path.join(root, '..', chgext(filename, '.txt'))):
                gtInimg = loadbbox(anno, 'xml')
                gt_sub = pd.DataFrame(gtInimg, columns=['class', *LTRB_cols, 'img_W', 'img_H'])
                gt_sub[LTRB_cols] = gt_sub[LTRB_cols]
                if isfile(filename):
                    gt_sub['img'] = filename
            else:
                raise FileNotFoundError

        elif form=='voc':
            # filename = pickFilename(filename)
            # anno = join(root, f'Annotations/{filename}.xml')
            anno = join(root, 'Annotations', filename)
            anno = chgext(anno, '.xml')
            if isfile(anno):
                gtInimg = loadbbox(anno, 'xml')
                gt_sub = pd.DataFrame(gtInimg, columns=['class', *LTRB_cols, 'img_W', 'img_H'])
                gt_sub[LTRB_cols] = gt_sub[LTRB_cols]
                gt_sub['img'] = filename
                # if isfile(join(root,f'JPEGImages/{filename}.jpg')):
                #     gt_sub['img'] = f'JPEGImages/{filename}.jpg'
                # elif isfile(join(root,f'JPEGImages/{filename}.png')):
                #     gt_sub['img'] = f'JPEGImages/{filename}.png'
            else:
                print(f'{anno}가 없습니다.')
                continue
        else:
            gt_sub = pd.DataFrame()

        gt = gt.append(gt_sub, ignore_index=False)
    return gt

def process_gt(file_list, subset:str, root, form='yolo', **kwargs):
    assert form in forms, '데이터셋 양식을 확인하세요.'
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
        gt_part[ltrb_cols] = (gt_part[ltrb_cols] * gt_part[['img_W','img_H']*2].values)
        return gt_part

    if form=='yolo':
        if gt['class'].dtype=='int':
            gt_part = process_txt(gt)
        else:
            gt_part = process_xml(gt)
        save_csv(gt_part, f'gtpart_{subset}.csv', root)

    elif form=='voc':
        gt_part = process_xml(gt)
        save_csv(gt_part, f'gtpart_{subset}.csv', root)
    else:
        gt_part = pd.DataFrame()



    return gt_part

def IOU(Bbox1, Bbox2):
    '''
    Bbox = (left, top, right, bottom)
    '''
    # intersection_x = max([0, ((Bbox1[2]-Bbox1[0])+(Bbox2[2]-Bbox2[0])-np.abs(Bbox1[0]-Bbox2[0])-np.abs(Bbox1[2]-Bbox2[2]))/2])
    # intersection_y = max([0, ((Bbox1[3]-Bbox1[1])+(Bbox2[3]-Bbox2[1])-np.abs(Bbox1[1]-Bbox2[1])-np.abs(Bbox1[3]-Bbox2[3]))/2])


    # Area1 = (Bbox1[2]-Bbox1[0])*(Bbox1[3]-Bbox1[1])
    # Area2 = (Bbox2[2]-Bbox2[0])*(Bbox2[3]-Bbox2[1])
    # intersectionArea = intersection_x*intersection_y

    # iou = intersectionArea / (Area1+Area2-intersectionArea)
    Bbox2 = Bbox2.transpose()
    intersection_x = np.maximum(0, ((Bbox1[:,[ 2 ]]-Bbox1[:,[ 0 ]]) \
        +(Bbox2[2,:]-Bbox2[0,:]) \
        -np.abs(Bbox1[:,[ 0 ]]-Bbox2[0,:]) \
        -np.abs(Bbox1[:,[ 2 ]]-Bbox2[2,:]))/2)
    intersection_y = np.maximum(0, ((Bbox1[:,[ 3 ]]-Bbox1[:,[ 1 ]]) \
        +(Bbox2[3,:]-Bbox2[1,:]) \
        -np.abs(Bbox1[:,[ 1 ]]-Bbox2[1,:]) \
        -np.abs(Bbox1[:,[ 3 ]]-Bbox2[3,:]))/2)


    Area1 = ((Bbox1[:,2]-Bbox1[:,0])*(Bbox1[:,3]-Bbox1[:,1])).reshape((-1,1))
    Area2 = (Bbox2[2,:]-Bbox2[0,:])*(Bbox2[3,:]-Bbox2[1,:])
    intersectionArea = intersection_x*intersection_y

    iou = intersectionArea / (Area1+Area2-intersectionArea)
    return iou

def to_perObj(pred_part, imgtxtpath, detection_time, IOU_thresh=0.5, conf_thresh=0.25, load_gt_part=None, root='', **kwargs):
    # namesfile = kwargs['namesfile']
    configPath = pickFilename(kwargs['configPath'])
    weight = pickFilename(kwargs['weightPath']).split('_')[-1]

    subset = pickFilename(imgtxtpath)
    # if load_gt_part==None:
    if not os.path.isfile(gtpart_path:=os.path.join(root, f'gtpart_{subset}.csv')):
        gt_part = process_gt(pred_part, subset, root, **kwargs)
    else:
        gt_part = load_csv(gtpart_path)
    matching_result = matching(pred_part, gt_part, IOU_thresh, conf_thresh)
    eval_condition(matching_result)
    if kwargs['form'] =='yolo':
        save_csv(matching_result, f'{subset}_perObj_{configPath}_{weight}_It{IOU_thresh}_ct{conf_thresh}_{detection_time}.csv', root)
    elif kwargs['form'] =='voc':
        save_csv(matching_result, f'{subset}_perObj_{configPath}_{weight}_It{IOU_thresh}_ct{conf_thresh}_{detection_time}.csv', root)
    return matching_result

def ar_val_co(iou):
    # argmax = np.argmax(iou, axis=1)[np.any(~np.isneginf(iou), axis=1)]
    argmax = np.argmax(iou, axis=1)
    argmax = np.where(np.any(~np.isneginf(iou), axis=1), argmax, -1)
    values, counts = np.unique(argmax[argmax!=-1], return_counts=True)
    return argmax, values, counts

def matching(pred_part, gt_part, IOU_thresh, conf_thresh):
    pred_ltrb = ['pred_left','pred_top','pred_right','pred_bottom']
    gt_ltrb = ['gt_left','gt_top','gt_right','gt_bottom']
    pred_part = pred_part.loc[pred_part['conf']>=conf_thresh]
    pred_part['IoU'] = np.nan
    pred_part['id'] = np.nan
    

    for img in pred_part['img'].unique():
        inter_classes = set(pred_part.loc[pred_part['img']==img, 'class'].unique()) & set(gt_part.loc[gt_part['img']==img, 'class'].unique())
        for cls in inter_classes:
            gt_bbox = gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cls), gt_ltrb]
            pred_bbox = pred_part.loc[(pred_part['img']==img) & (pred_part['class']==cls), pred_ltrb]
            iou = IOU(pred_bbox.to_numpy(), gt_bbox.to_numpy())
            # row_condition = (pred_part['img']==img) & (pred_part['class']==cls)
            # for index,value in gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cls)].iterrows():
            #     GTBbox = value[gt_ltrb]
            #     iou = pred_part.loc[row_condition, pred_ltrb].apply(lambda predBbox: IOU(GTBbox, predBbox),axis=1)
            #     if not (iou.count() == 0).all():
            #         iouidx = iou.idxmax()
            #         ioumax = iou.max()
            #         if ((IOU_thresh <= ioumax) & (ioumax > pred_part.loc[row_condition].fillna(-1).loc[iouidx, 'IoU'].item())):
            #             pred_part.loc[iouidx, 'id'] = gt_part.loc[index,'id']
            #             pred_part.loc[iouidx, 'IoU'] = ioumax
            argmax, values, counts = ar_val_co(iou)
            while (counts_max:=np.max(counts))>1:
                duple_idx=np.max(values[counts==counts_max])
                iou[(argmax==duple_idx)&(np.arange(argmax.size)!=np.argmax(iou[:, duple_idx])),duple_idx] = -np.inf
                argmax, values, counts = ar_val_co(iou)
            iou = np.max(iou, axis=1)
            iou[iou<IOU_thresh] = np.nan
            pred_part.loc[(pred_part['img']==img) & (pred_part['class']==cls), 'IoU'] = iou
            gt_id = gt_part.loc[(gt_part['img']==img) & (gt_part['class']==cls), 'id']
            pred_part.loc[(pred_part['img']==img) & (pred_part['class']==cls), 'id'] = np.where(~np.isnan(iou), gt_id.values[argmax], np.nan)

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
    will_add_evals_col = list(filter(lambda eval_elm: eval_elm not in evals.columns.tolist(), ['TP', 'FP', 'FN']))
    for add_col in will_add_evals_col:
        evals[add_col] = np.nan
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
            img_union_count_perDataset = perOther['perImg'][['class', 'img']].groupby('class').count().rename(columns={'img':'img_union_count'})
            img_gt_count_perDataset = perOther['perImg'].loc[perOther['perImg']['gt_count']!=0, ['class', 'img']].groupby('class').count().rename(columns={'img':'img_gt_count'})
            img_pred_count_perDataset = perOther['perImg'].loc[perOther['perImg']['pred_count']!=0, ['class', 'img']].groupby('class').count().rename(columns={'img':'img_pred_count'})
            runtime_perDataset = perOther['perImg'][['class', 'runtime']].groupby('class').agg(numeric_cols+['sum'])
            chg_colname(runtime_perDataset)
            img_size_perDataset = perOther['perImg'][['class','img_W', 'img_H']].groupby('class').agg(numeric_cols)
            chg_colname(img_size_perDataset)
            etc_perDataset = pd.concat([img_gt_count_perDataset, img_pred_count_perDataset, img_union_count_perDataset, runtime_perDataset, img_size_perDataset], axis=1).reset_index().rename(columns={'index':'class'})
            perOther_table = pd.merge(perOther_table, etc_perDataset)
        elif other=='perImg':
            perOther_table.rename(columns={'runtime_max':'runtime','img_W_max':'img_W','img_H_max':'img_H'}, inplace=True)

        perOther[other] = perOther_table

    for other, table in perOther.items():

        recall = (table['TP_count'] / (table['TP_count']+table['FN_count'])).rename('recall')
        precision = (table['TP_count'] / (table['TP_count']+table['FP_count'])).rename('precision')
        F1 = (2*recall*precision / (recall+precision)).rename('F1-Score')

        table = pd.concat([table, recall, precision, F1], axis=1)
        save_csv(table, f'{subset}_{other}_{configPath}_{weight}_It{IOU_thresh}_ct{conf_thresh}_{detection_time}.csv', root)

def run(pred_file):
    pred_file_split = pickFilename(pred_file).split('_')
    if re.match('.*[/]VOC[0-9/]{5}.*', pred_file):
        form='voc'
    else:
        form='yolo'
    arg = { 'IOU_thresh' : 0.5,
            'conf_thresh' : 0.25,
            'imgtxtpath' : f'{pred_file_split[0]}.txt',
            'configPath' : f"{pred_file_split[2]}.cfg", 
            'weightPath' : f"{pred_file_split[3]}.weights", 
            'detection_time' : pred_file_split[4],
            'root': os.path.join(os.path.split(pred_file)[0]),
            'form': form}
    arg['namesfile'] = f"{arg['root']}/obj.names"

    pred = load_csv(pred_file)
    perObj_predpart = process_pred(pred, **arg)
    perObj = to_perObj(perObj_predpart, **arg)
    to_other(perObj, **arg)

if __name__ == "__main__":
    #pred csv file
    pred_file = os.path.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_20200711_Meta-RCNN_1/VOC2007/trainval_pred_MRCN_1-100-263_200717-1449.csv')
    run(pred_file)
    
    # defect_work_path = os.path.expanduser('~/nasrw/mk/work_dataset')
    # defect_work_dirs = glob(os.path.expanduser('~/nasrw/mk/work_dataset/2DOD_defect_20200713_Meta-RCNN_1'))

    # pred_files = []
    # for dir in defect_work_dirs:
    #     for p, ds, fs in os.walk(dir):
    #         if fs:
    #             for f in fs:
    #                 if re.match('.*_pred_.*csv', f):
    #                     pred_files.append(os.path.join(p,f))
    
    # with Pool() as p:
    #     list(tqdm(p.imap(run, pred_files), total=len(pred_files)))