from collections import defaultdict
from pandas.core import base
import schedule
from typing import DefaultDict
from dateutil.tz.tz import gettz
import pandas as pd
import asyncio as aio
import requests
from functools import partial
from tqdm.asyncio import tqdm
from copy import Error, deepcopy
import datetime as dt
from itertools import cycle
import numpy as np
import json
from easydict import EasyDict
import json
from getpass import getpass
import gspread as gs
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone


#### FUNCTIONS ###
def getBaseUrl():
    if cfg.base_url.endswith('/'):
        base_url = '{}api/v1/'.format(cfg['base_url'])
    else:
        base_url = '{}/api/v1/'.format(cfg['base_url'])
    return base_url

def getToken():
    while True:
        print("Getting token...")
        # api_login = input('id: ')
        # api_password = getpass('pass: ')
        data_get = {'username': cfg.username,
                    'password': cfg.password}

        if cfg['base_url'].endswith('/'):
            base_url = '{}api/v1/'.format(cfg['base_url'])
        else:
            base_url = '{}/api/v1/'.format(cfg['base_url'])

        try:
            r = requests.post(base_url + 'auth/login', data=data_get, timeout=3)
            checkConnect = r.ok
        except :
            checkConnect = False
            print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

        if checkConnect:
            token = r.json()['key']
            cfg['token'] = token
            saveJson(cfg)
            print('token이 config파일에 저장되었습니다. ')
            return token

        else:
            print('로그인 실패')

def tokenCheck():
    global cfg, header_gs
    print('token check...',end=None)
    r = requests.get(base_url+f'users/self', headers=header_gs)
    if (not r.ok):
        if r.reason == 'Unauthorized':
            getToken()
            cfg = EasyDict(readJson())
            header_gs = {'Accept': 'application/json', 'Authorization': f'Token {cfg.token}'}
        else:
            print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
    else:
        print('{} complete'.format(dt.datetime.now(gettz('Asia/Seoul'))))

def readJson():
    with open(path,'r') as f:
        cfg = json.load(f)
    return cfg

def saveJson(jsonDict):
    with open(path, 'w') as f:
        json.dump(jsonDict, f,ensure_ascii=False, indent=4)

def getTasksInfo():
    global taskInfo
    r = requests.get(base_url+f'projects/{cfg.projectId}', headers=header_gs)
    if r.ok:
        projectInfo = r.json()
        projectName = projectInfo['name']
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
        raise ConnectionError

    r = requests.get(base_url+f'projects/{cfg.projectId}/tasks', headers=header_gs)
    r = requests.get(base_url+f'projects/{cfg.projectId}/tasks', headers=header_gs, params={'page_size':r.json()['count']})
    if r.ok:
        taskName = dict()
        labels = dict()
        jobUrls = dict()
        jobSize = dict()
        jobAssignee = dict()
        jobStatus = dict()
        jobTaskId = dict()
        keys = []
        taskFrameCount = dict()

        for task in r.json()['results']:
            # meta = await getTaskMeta(task['url'], header_gs)
            taskFrameCount[task['id']] = task['size']

            key = [projectName,task['id'], task['name']]
            taskName[task['id']] = task['name']
            for label in task['labels']:
                labels[label['id']] = label['name']
            for job in task['segments']:
                jobSize[job['jobs'][0]['id']] = job['stop_frame']-job['start_frame']+1
                jobUrls[job['jobs'][0]['id']] = job['jobs'][0]['url']
                jobAssignee[job['jobs'][0]['id']] = job['jobs'][0]['assignee']
                jobStatus[job['jobs'][0]['id']] = job['jobs'][0]['status']
                jobTaskId[job['jobs'][0]['id']] = task['id']
                _key = key+[job['jobs'][0]['id']]
                keys.append(_key)

        taskInfo = {'taskName': taskName,
                  'labels': labels,
                  'jobUrls': jobUrls,
                  'taskFrameCount': taskFrameCount,
                  'jobSize': jobSize,
                  'jobAssignee': jobAssignee,
                  'jobStatus': jobStatus,
                  'jobTaskId': jobTaskId,
                  'keys': keys}

        # Task name에서 type과 class 얻기
        taskInfo['jobLabelType'] = dict()
        taskInfo['jobLabelClass'] = dict()
        for jobId, _ in taskInfo['jobSize'].items():
            _, labelType, labelClass = taskInfo['taskName'][taskInfo['jobTaskId'][jobId]].split('_')
            taskInfo['jobLabelType'][jobId] = labelType
            taskInfo['jobLabelClass'][jobId] = labelClass

        print('{} get task info'.format(dt.datetime.now(gettz('Asia/Seoul'))))
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
        raise ConnectionError

def getLabelerInfo():
    global labelerInfo
    print('get labler')
    labelerInfo = pd.DataFrame(gc.open('log').worksheet('라벨러').get_all_records()).set_index('id')
    print('{} get labeler info'.format(dt.datetime.now(gettz('Asia/Seoul'))))

async def getAnnotationInfo(jobUrl):
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs)
    r = await loop.run_in_executor(None, get, jobUrl + "/annotations")
    if r.ok:
        return r
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def mkMonitoringFile():
    getTasksInfo()
    aio_tasks = dict()
    datas = []
    async def getData(key):
        url = taskInfo['jobUrls'][key[3]]
        annotations = (await getAnnotationInfo(url)).json()
        for anno in annotations['tags']:
            _key = list(key)
            _key.append('tags')
            _key.append(anno['frame'])
            _key.append(taskInfo['labels'][anno['label_id']])
            datas.append(_key)
        for anno in annotations['shapes']:
            _key = list(key)
            _key.append(anno['type'])
            _key.append(anno['frame'])
            _key.append(taskInfo['labels'][anno['label_id']])
            datas.append(_key)
        for anno in annotations['tracks']:
            for shape in anno['shapes']:
                _key = list(key)
                _key.append(shape['type'])
                _key.append(anno['frame'])
                _key.append(taskInfo['labels'][anno['label_id']])
                datas.append(_key)

    aio_tasks = []
    for key in taskInfo['keys']:
        aio_tasks.append(aio.create_task(getData(key)))

    async for aio_task in tqdm(aio_tasks, desc='jobs'):
        await aio_task

    cols = ['project','task_id','task_name','job_id','label_type','frame','class']
    df = pd.DataFrame(datas, columns=cols)
    df['project'] = df['project'].fillna('default')

    # 각 클래스에 해당하는 오브젝트 수
    dfCountObjectJobClass = df[['project','task_id','task_name','job_id','label_type','class','frame']]\
                            .groupby(['project','task_id','task_name','job_id','class','label_type']).count().unstack(-2)
    dfCountObjectJobClass.columns = dfCountObjectJobClass.columns.droplevel(level=0)

    dfCountObjectTaskClass = df[['project','task_id','task_name','job_id','label_type','class']]\
                            .groupby(['project','task_id','task_name','class','label_type']).count().unstack(-2)
    dfCountObjectTaskClass.columns = dfCountObjectTaskClass.columns.droplevel(level=0)

    dfCountObjectProjClass = dfCountObjectTaskClass.groupby(['project','label_type']).sum()

    # 라벨링 된 이미지 수
    dfCountImgJob = df[['project','task_id','task_name','job_id','frame']].drop_duplicates(['project','task_id','job_id','frame'])\
                    .groupby(['project','task_id','task_name','job_id']).count()
    dfCountImgJob.columns = ['isin']
    dfCountImgJob['image_size'] = dfCountImgJob.index.droplevel(level=[0,1,2]).map(taskInfo['jobSize'])
    dfCountImgJob['not isin'] = dfCountImgJob['image_size']-dfCountImgJob['isin']

    dfCountImgTask = df[['project','task_id','task_name','frame']].drop_duplicates(['project','task_id','frame'])\
                    .groupby(['project','task_id','task_name']).count()
    dfCountImgTask.columns = ['isin']
    dfCountImgTask['image_size'] = dfCountImgTask.index.droplevel(level=[0,-1]).map(taskInfo['taskFrameCount'])
    dfCountImgTask['not isin'] = dfCountImgTask['image_size']-dfCountImgTask['isin']

    dfCountImgProj = dfCountImgTask.groupby(['project']).sum()

    # 각 클래스에 해당하는 이미지 수
    dfCountImgJobClass = df[['project','task_id','task_name','job_id','label_type','class','frame']]\
                         .drop_duplicates(['project','task_id','task_name','job_id','label_type','class','frame']) \
                         .groupby(['project','task_id','task_name','job_id','class','label_type']).count().unstack(-2)
    dfCountImgJobClass.columns = dfCountImgJobClass.columns.droplevel(level=0)

    dfCountImgTaskClass = df[['project','task_id','task_name','label_type','class','frame']]\
                         .drop_duplicates(['project','task_id','task_name','label_type','class','frame']) \
                         .groupby(['project','task_id','task_name','class','label_type']).count().unstack(-2)
    dfCountImgTaskClass.columns = dfCountImgTaskClass.columns.droplevel(level=0)

    dfCountImgProjClass = dfCountImgTaskClass.groupby(['project','label_type']).sum()

    # Save
    time = dt.datetime.now(gettz('Asia/Seoul')).strftime(timeFormat)
    dfCountObjectJobClass.to_csv(f'./{time}_CVAT-Object-Class-Job.csv', encoding='euc-kr')
    dfCountObjectTaskClass.to_csv(f'./{time}_CVAT-Object-Class-Task.csv', encoding='euc-kr')
    dfCountObjectProjClass.to_csv(f'./{time}_CVAT-Object-Class-Proj.csv', encoding='euc-kr')
    dfCountImgJob.to_csv(f'./{time}_CVAT-Image-Isin-Job.csv', encoding='euc-kr')
    dfCountImgTask.to_csv(f'./{time}_CVAT-Image-Isin-Task.csv', encoding='euc-kr')
    dfCountImgProj.to_csv(f'./{time}_CVAT-Image-Isin-Proj.csv', encoding='euc-kr')
    dfCountImgJobClass.to_csv(f'./{time}_CVAT-Image-Class-Job.csv', encoding='euc-kr')
    dfCountImgTaskClass.to_csv(f'./{time}_CVAT-Image-Class-Task.csv', encoding='euc-kr')
    dfCountImgProjClass.to_csv(f'./{time}_CVAT-Image-Class-Proj.csv', encoding='euc-kr')

def registUser():
    global labelerInfo
    sheet = gc.open('log').worksheet('라벨러')
    labelerInfo = pd.DataFrame(sheet.get_all_records())
    cols = labelerInfo.columns
    labelerInfo = labelerInfo.set_index('username')
    registList = labelerInfo.loc[~labelerInfo['id'].map(bool)]
    for labeler in registList.itertuples():
        data = {
                "username": labeler.Index,
                # "email": labeler.email,
                "password1": labeler.password,
                "password2": labeler.password,
                "first_name": labeler.first_name,
                "last_name": labeler.last_name,
                }
        requests.post(base_url + "auth/register", data=data)
    r = requests.get(base_url+"users", headers=header_gs)
    userCount = r.json()['count']
    r = requests.get(base_url+"users", headers=header_gs, params={'page_size': userCount})
    for user in r.json()['results']:
        username = user['username']
        if username in labelerInfo.index.values:
            labelerInfo.loc[username,'id'] = user['id']
    labelerInfo.reset_index(inplace=True)
    labelerInfo = labelerInfo.reindex(columns= cols)
    sheet.clear()
    sheet.update([labelerInfo.columns.values.tolist()]+labelerInfo.values.tolist())



# def assignLabeler(newAssignmentSize=1000):
#     taskInfo = getTasksInfo()
#     eachLabelerCR = getClassCR(newAssignmentSize)
#     assign(eachLabelerCR)
#     updateJobAssign()
#     logWorkload(eachLabelerCR)

def assignment2Df(assignCR:dict, classCRProj, onceAssignWorkingHour, hourlyWorkingCount):
    seriesList = []
    for key, val in assignCR.items():
        seriesList.append(pd.Series(val,name=key))

    classAsnDf = pd.DataFrame(seriesList).T.fillna(0)
    classAsnDf = classAsnDf.reindex(classCRProj.index, fill_value=0)
    classAsnDf['Rate'] = classAsnDf['assignment']/classAsnDf['assignment'].sum()
    newAssignmentTotal = onceAssignWorkingHour-(classAsnDf['annotation']/hourlyWorkingCount).sum()
    classAsnDf['newAssignmentTarget'] = ((classAsnDf['assignment']/hourlyWorkingCount).sum()+newAssignmentTotal)*classCRProj['Rate']-(classAsnDf['assignment']/hourlyWorkingCount)
    return {"newAssignmentSize":newAssignmentTotal, 'data':classAsnDf}

def getAssignCR():
    global assignCR
    getTasksInfo()

    #project 전체의 이미지 수, 각 class의 비율
    classCountProjBbox = DefaultDict(int)
    classCountProjSeg = DefaultDict(int)
    for taskId, count in taskInfo['taskFrameCount'].items():
        _, labelType, labelClass = taskInfo['taskName'][taskId].split('_')
        if labelType=='Bbox':
            classCountProjBbox[f'{labelClass}'] += count
        elif labelType=='Seg':
            classCountProjSeg[f'{labelClass}'] += count
        else:
            raise TypeError
    classCountProjBbox = pd.Series(classCountProjBbox)
    classRateProjBbox = classCountProjBbox/classCountProjBbox.sum()
    classCRProjBbox = pd.DataFrame([classCountProjBbox,classRateProjBbox], index=['assignment', 'Rate']).T
    
    classCountProjSeg = pd.Series(classCountProjSeg)
    classRateProjSeg = classCountProjSeg/classCountProjSeg.sum()
    classCRProjSeg = pd.DataFrame([classCountProjSeg,classRateProjSeg], index=['assignment', 'Rate']).T
    

    #assignee가 할당된 job의 이미지 수, 각 class의 비율
    classCountAsnlist = lambda : {'Bbox': {'assignment':DefaultDict(int),
                                           'annotation':DefaultDict(int), 
                                           'validation':DefaultDict(int), 
                                           'modification':DefaultDict(int), 
                                           'completed':DefaultDict(int)},
                                  'Seg' : {'assignment':DefaultDict(int),
                                           'annotation':DefaultDict(int), 
                                           'validation':DefaultDict(int), 
                                           'modification':DefaultDict(int), 
                                           'completed':DefaultDict(int)}}
    assignCR = DefaultDict(classCountAsnlist)
    for jobId, count in taskInfo['jobSize'].items():
        jobAssignee = taskInfo['jobAssignee'][jobId]
        _, labelType, labelClass = taskInfo['taskName'][taskInfo['jobTaskId'][jobId]].split('_')
        if jobAssignee != None:
            if labelType=='Bbox':
                assignCR[jobAssignee]['Bbox']['assignment'][labelClass] += count
                status =  taskInfo['jobStatus'][jobId]
                assignCR[jobAssignee]['Bbox'][status][labelClass] += count
            elif labelType=='Seg':
                assignCR[jobAssignee]['Seg']['assignment'][labelClass] += count
                status =  taskInfo['jobStatus'][jobId]
                assignCR[jobAssignee]['Seg'][status][labelClass] += count
            else:
                raise TypeError
    newAssignCR = DefaultDict(dict)
    segActivate = labelerInfo.loc[labelerInfo['Seg 희망'].map(bool)].index
    boxActivate = labelerInfo.loc[labelerInfo['Bbox 희망'].map(bool)].index
    for assignee in labelerInfo.index:
        newAssignCR[assignee]['Bbox'] = assignment2Df(assignCR[assignee]['Bbox'],classCRProjBbox, cfg.onceAssignWorkingHour,cfg.hourlyBboxWorkingCount)
        newAssignCR[assignee]['Seg'] = assignment2Df(assignCR[assignee]['Seg'], classCRProjSeg, cfg.onceAssignWorkingHour, cfg.hourlySegWorkingCount)

        if assignee not in boxActivate:
            newAssignCR[assignee]['Bbox']['newAssignmentSize'] = 0
        if assignee not in segActivate:
            newAssignCR[assignee]['Seg']['newAssignmentSize'] = 0

    assignCR = newAssignCR

def getDailyWorkingHour():
    global dailyWorkingHour
    dailyWorkingHour = dict([(assignee, cfg.dailyWorkingHour) for assignee in labelerInfo['username'].keys()])

def getNewAssignDict(labelType):
    getTasksInfo()
    global dailyWorkingHour
    newAssignDict = dict()

    unassignJobList = [jobId for jobId, assignee in taskInfo['jobAssignee'].items() if assignee==None if taskInfo['jobLabelType'][jobId]==labelType]
    unassignClassJobDict = defaultdict(list)
    for jobId in unassignJobList:
        unassignClassJobDict[taskInfo['jobLabelClass'][jobId]].append(jobId)

    for labeler, data in assignCR.items():
        if (labeler in labelerInfo.loc[labelerInfo['activate'].map(lambda x: bool(str(x).strip())) & labelerInfo[f'{labelType} 희망'].map(lambda x: bool(str(x).strip()))].index):
            while (data[labelType]['newAssignmentSize']>0)&(dailyWorkingHour[labeler]>0):
                assignClass = data[labelType]['data'].loc[unassignClassJobDict.keys(), 'newAssignmentTarget'].idxmax()
                assignJobId = unassignClassJobDict[assignClass].pop()
                newAssignDict[assignJobId] = int(labeler)
                if labelType == 'Bbox':
                    hourlyWorkingCount = cfg.hourlyBboxWorkingCount
                elif labelType == 'Seg':
                    hourlyWorkingCount = cfg.hourlySegWorkingCount
                else:
                    raise Error
                data['Bbox']['newAssignmentSize'] -= taskInfo['jobSize'][assignJobId]/cfg.hourlyBboxWorkingCount
                data['Seg']['newAssignmentSize'] -= taskInfo['jobSize'][assignJobId]/cfg.hourlySegWorkingCount
                data[labelType]['data'].loc[assignClass,'newAssignmentTarget'] -= taskInfo['jobSize'][assignJobId]/hourlyWorkingCount
                dailyWorkingHour[labeler] -= taskInfo['jobSize'][assignJobId]/hourlyWorkingCount
                taskInfo['jobAssignee'][assignJobId] = labeler
    return newAssignDict


getJobUrl = lambda taskId, jobId: f'{cfg.base_url}tasks/{taskId}/jobs/{jobId}' \
                                  if cfg.base_url.endswith('/') \
                                  else f'{cfg.base_url}/tasks/{taskId}/jobs/{jobId}'

def assign(newAssignDict):
    for jobId, assignee in newAssignDict.items():
        requests.patch(base_url + f"jobs/{jobId}", headers=header_gs, data={"assignee": int(assignee)})
    getTasksInfo()
    print('{} assign'.format(dt.datetime.now(gettz('Asia/Seoul'))))

def getAssignTable():
    getTasksInfo()
    keysCols = ['project','task_id','task_name','job_id']
    assignTable = pd.DataFrame(taskInfo['keys'], columns=keysCols)
    assignTableCols = ['jobSize', 'jobAssignee', 'jobStatus']
    for col in assignTableCols:
        assignTable[col] = assignTable['job_id'].map(taskInfo[col])
    assignTable['jobUrl'] = assignTable[['task_id','job_id']].apply(lambda x: getJobUrl(x.task_id,x.job_id), axis=1)
    assignTable['assigneeName'] = assignTable['jobAssignee'].map(labelerInfo['username'])
    assignTable['assignTime'] = dt.datetime.now(gettz('Asia/Seoul')).strftime(timeFormat)
    assignTable['inCharge'] = assignTable['jobAssignee'].map(labelerInfo['inCharge'])
    return assignTable

def updateJobAssign(assignTable):
    cols = ['assignTime','project','task_id','task_name','job_id','jobUrl', 'jobSize', 'jobAssignee','assigneeName', 'inCharge', 'jobStatus']
    assignTable = assignTable.reindex(columns=cols).set_index('job_id').dropna()

    sheet = gc.open('job 할당표').worksheet('job 할당표')
    preAssignTable = pd.DataFrame(sheet.get_all_records()).reindex(columns=cols)
    if preAssignTable['inCharge'].isna().all():
        preAssignTable['inCharge'] = preAssignTable['jobAssignee'].map(labelerInfo['inCharge'])
    preAssignTable.set_index('job_id',inplace=True)

    assignTable['assignTime'].update(preAssignTable['assignTime'])
    assignTable = assignTable.reset_index().reindex(columns=cols)
    sheet.clear()
    sheet.update([assignTable.columns.values.tolist()]+assignTable.values.tolist())
    print('{} update job assign'.format(dt.datetime.now(gettz('Asia/Seoul'))))

def logWorkload():
    getTasksInfo()
    getAssignCR()
    exportSum = []
    exportClass = []
    labelerSumIndexNames = ['time','assigneeId','assigneeName','inCharge','labelType']
    labelerClassIndexNames = ['time','assigneeId','assigneeName','inCharge', 'labelType', 'labelClass']
    projectSumIndexNames = ['time', 'labelType']
    projectClassIndexNames = ['time', 'labelType', 'labelClass']
    cols = ['assignment', 'annotation', 'validation', 'modification', 'completed']
    totalCols = [f'total {col}' for col in cols]
    gradCols = [f'grad {col}' for col in cols]
    execTime = dt.datetime.now(gettz('Asia/Seoul')).strftime(timeFormat)
    for assignee, labelType in assignCR.items():
        for type, data in labelType.items():
            classDf = data['data'].reindex(columns=cols)
            classDf.columns = totalCols
            classDf = classDf.reindex(index=list(set(taskInfo['labels'].values())), fill_value=0)
            sumDf = classDf.sum(axis=0)
            sumDf['assigneeId'] = classDf['assigneeId'] = assignee
            sumDf['assigneeName'] = classDf['assigneeName'] = labelerInfo['username'][assignee]
            sumDf['inCharge'] = classDf['inCharge'] = labelerInfo['inCharge'][assignee]
            sumDf['labelType'] = classDf['labelType'] = type
            sumDf['time'] = classDf['time'] = execTime

            exportSum.append(sumDf)
            exportClass.append(classDf.reset_index().rename(columns={'index':'labelClass'}))
    labelerTotalWorkloadClass = pd.concat(exportClass).set_index(labelerClassIndexNames).reindex(columns=totalCols)
    labelerTotalWorkloadSum = pd.DataFrame(exportSum).set_index(labelerSumIndexNames).reindex(columns=totalCols)

    # totalCount = taskInfo['name'].str.split('_')
    projectTotalWorkloadClass = labelerTotalWorkloadClass.groupby(projectClassIndexNames).sum()
    projectTotalWorkloadSum = labelerTotalWorkloadSum.groupby(projectSumIndexNames).sum()


    def getNewWorkload(totalWorkload, IndexNames, sheetName):
        sheet = gc.open('log').worksheet(sheetName)
        preWorkload = pd.DataFrame(sheet.get_all_records())
        if not preWorkload.empty:
            preWorkload = preWorkload.reindex(columns=IndexNames+totalCols+gradCols)
            if 'inCharge' in preWorkload.columns.values:
                if preWorkload['inCharge'].isna().any():
                    preWorkload.loc[preWorkload['inCharge'].isna(),'inCharge'] = preWorkload.loc[preWorkload['inCharge'].isna(),'assigneeId'].map(labelerInfo['inCharge'])
                    sheet.clear()
                    sheet.update([preWorkload.columns.values.tolist()] + preWorkload.values.tolist())
            preWorkload = preWorkload.set_index(IndexNames)
        else:
            preWorkload = pd.DataFrame(preWorkload, columns=IndexNames+totalCols+gradCols).set_index(IndexNames)
            preWorkload = preWorkload.reindex(index=totalWorkload.index, fill_value=0)
            
        #최근 작업량 구하기
        recentTime = preWorkload.index.get_level_values('time').map(lambda x: dt.datetime.strptime(x,timeFormat)).max().strftime(timeFormat)
        recentWorkload = preWorkload.loc[preWorkload.index.get_level_values('time')==recentTime,totalCols].copy(deep=True)
        recentWorkload.index = pd.MultiIndex.from_tuples([(execTime, *r[1:]) for r in list(recentWorkload.index)])
        recentWorkload = recentWorkload.reindex(index=totalWorkload.index, fill_value=0)

        # 새 작업량 구하기
        newWorkload = totalWorkload.reindex(columns=totalCols, fill_value=0)
        gradWorkload = newWorkload - recentWorkload
        gradWorkload.columns = gradCols
        newWorkload = pd.concat([totalWorkload, gradWorkload], axis=1)

        #sheet 업데이트
        if recentTime == execTime:
            sheet.update([newWorkload.reset_index().columns.values.tolist()] + newWorkload.reset_index().values.tolist())
        else:
            sheet.insert_rows(newWorkload.reset_index().values.tolist(),row=2)

    getNewWorkload(labelerTotalWorkloadSum,labelerSumIndexNames, '작업량(작업자-클래스합)')
    getNewWorkload(labelerTotalWorkloadClass,labelerClassIndexNames, '작업량(작업자-클래스별)')
    getNewWorkload(projectTotalWorkloadSum,projectSumIndexNames, '작업량(전체-클래스합)')
    getNewWorkload(projectTotalWorkloadClass,projectClassIndexNames, '작업량(전체-클래스별)')

    print('{} logging workload'.format(dt.datetime.now(gettz('Asia/Seoul'))))

def runAssign():
    getAssignCR()
    assign(getNewAssignDict('Seg'))
    assign(getNewAssignDict('Bbox'))
    print(dailyWorkingHour)

async def mainSelect():
    choice = None
    while choice != "0":
        getTasksInfo()
        print \
        ("""
        ---MENU---
        
        0 - Exit
        1 - Run
        2 - User Register
        3 - Get new assign dict
        4 - Update job assign table
        5 - Update log
        6 - Assign
        """)

        choice = input("Your choice: ") # What To Do ???
        print()
    
        if choice == "0":
            print("Good bye!")  
        elif choice == "1":
            await mkMonitoringFile()
        elif choice == "2":
            registUser()
        elif choice == "3":
            getDailyWorkingHour()
            getAssignCR()
            newAssignDict = getNewAssignDict('Bbox')
        elif choice == "4":
            updateJobAssign(getAssignTable())
        elif choice == "5":
            logWorkload()
        elif choice == "6":
            assign(newAssignDict)
        else:
            print(" ### Wrong option ### ")

async def mainSchedule():
    sched = BackgroundScheduler()
    sched.start()
    sched.add_job(logWorkload, 'cron', hour=15, minute=30, timezone=timezone('Asia/Seoul'), id=f"logging")
    sched.add_job(runAssign,'interval', hours=1, timezone=timezone('Asia/Seoul'), id=f"assign", next_run_time=startTime)
    sched.add_job(getTasksInfo,'interval', minutes=30, timezone=timezone('Asia/Seoul'), id=f"get task info")
    sched.add_job(getLabelerInfo,'interval', minutes=30, timezone=timezone('Asia/Seoul'), id=f"get labeler info")
    sched.add_job(getDailyWorkingHour, 'cron', hour=15, minute=30, timezone=timezone('Asia/Seoul'), id=f"init daily working hour")
    # updateJobTable = lambda : updateJobAssign(getAssignTable())
    sched.add_job(lambda : updateJobAssign(getAssignTable()), 'interval', minutes=5, id="update Job Assign", timezone=timezone('Asia/Seoul'))
    while True:
        schedule.run_pending()

class Project():
    def __init__(self) -> None:
        self.seledProjectId = cfg.projectId
        self.getTasksInfo()

    def mkProject(self,projName):
        requests.post(base_url+'projects', headers=header_gs, data={'name':projName})
    
    def getProjList(self):
        r = requests.get(base_url+'projects', headers=header_gs)
        r = requests.get(base_url+'projects', headers=header_gs, params={'page_size': r.json()['count']})
        print('no\tid\tname')
        print('----------------------------')
        print(f'0\t-\tall')
        projrepo = []
        for idx, proj in enumerate(r.json()['results']):
            num = idx +1
            print( f'{num}\t{proj["id"]}\t{proj["name"]}')
            projrepo.append(pd.Series(proj, name=num))
        self.projDf = pd.DataFrame(projrepo)

    def selProj(self):
        self.getProjList()
        print()
        selNum = int(input('no. 선택: '))
        try:
            if selNum == 0:
                self.seledProjectId = 'all'
            else:
                self.seledProj = self.projDf.loc[int(selNum)]
                self.seledProjectId = self.seledProj['id']
        except KeyError as e:
            print('잘못된 숫자를 선택했습니다.')
            self.selProj()

    def getTasksInfo(self):
        if self.seledProjectId == 'all':
            r = requests.get(base_url+f'tasks', headers=header_gs)
            r = requests.get(base_url+f'tasks', headers=header_gs, params={'page_size':r.json()['count']})
        else:
            r = requests.get(base_url+f'projects/{cfg.projectId}/tasks', headers=header_gs)
            r = requests.get(base_url+f'projects/{cfg.projectId}/tasks', headers=header_gs, params={'page_size':r.json()['count']})
        taskrepo = []
        for task in r.json()['results']:
            taskrepo.append(pd.Series(task))
        self.taskInfo = pd.DataFrame(taskrepo)

    def exportProjTaskTable(self):
        self.getProjList()
        table = self.taskInfo[['id', 'name', 'project']]\
                .rename(columns=dict(zip(['id', 'name', 'project'],['taskId', 'taskName', 'projId'])))
        table['projName'] = table['projId'].map(self.projDf[['id','name']].set_index('id').to_dict()['name'])
        table.to_csv(cfg.projectFile,encoding='euc-kr')

    def readProjTaskTable(self):
        '''
        tabel cols = [taskId, taskName, projId, projName]
        '''
        self.projTaskTable = pd.read_csv(cfg.projectFile)

    def assignProjFromTable(self):
        self.readProjTaskTable()
        for task in self.projTaskTable.itertuples():
            requests.patch(base_url+f'tasks/{task.taskId}', headers=header_gs, data={"project":task.projId})

    def assignProjAllUnassignedTask(self):
        self.selProj()
        unassignedTaskInfo = self.taskInfo.loc[self.taskInfo['project'].isna()]
        if self.seledProjectId == 'all':
            print('다른 Project를 선택해 주세요')
            self.selProj()
        for info in unassignedTaskInfo.itertuples():
            requests.patch(base_url+f'tasks/{info.taskId}', headers=header_gs, data={"project":self.seledProj})




    # def setProj(self):


if __name__ == "__main__":
    debug=False
    timeFormat = '%y%m%d-%H%M'
    path = 'config.json'
    cfg = EasyDict(readJson())
    header_gs = {'Accept': 'application/json', 'Authorization': f'Token {cfg.token}'}
    base_url = getBaseUrl()
    startTime = dt.datetime.now(timezone('Asia/Seoul')).replace(microsecond=0,minute=0,second=0) + dt.timedelta(hours=1)
    tokenCheck()
    getTasksInfo()
    gc = gs.service_account(filename='nia-dataset-83bf2b5f03fd.json')
    getLabelerInfo()
    getDailyWorkingHour()
    # dailyWorkingHour ={4: 6.0, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 9.0, 11: 4.0, 12: 7.0, 13: -1.0, 14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 9.0, 22: 0.0, 23: 8.0, 24: 0.0, 25: 10, 26: 9.0, 27: 5.0, 28: 10, 44: 10, 47: 5.0, 45: 1.0, 46: 4.0}
    # runAssign()
    # tt = Project()
    # tt.assignProjAllUnassignedTask()
    if debug:
        aio.run(mainSelect())
    else:
        aio.run(mainSchedule())