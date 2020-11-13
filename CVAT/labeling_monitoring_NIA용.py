import os
from typing import DefaultDict
from dateutil.tz.tz import gettz
import pandas as pd
import asyncio as aio
import requests
from functools import partial
from getpass import getpass
from tqdm.asyncio import tqdm
from copy import deepcopy
import datetime as dt
from itertools import cycle
import datetime as dt
import numpy as np

#### FUNCTIONS ###
def auth():
    if not debug:
        api_login = input('id: ')
        api_password = getpass('pass: ')
        port = input('port: ')
    else:
        api_login = 'serveradmin'
        print(api_login)
        api_password = 'wnrWkd131@Cv'
        port = 12280
    return api_login, api_password, port

def login():
    while True:
        print("Getting token...")
        api_login,api_password, port = auth()
        base_url = f"http://tmlabel.duckdns.org:{port}/api/v1/"
        data_get = {'username': api_login,
                    'password': api_password}
        try:
            r = requests.post(base_url + 'auth/login', data=data_get, timeout=3)
            checkConnect = r.ok
        except:
            checkConnect = False

        if checkConnect:
            token = r.json()['key']
            header_gs = {'Accept': 'application/json', 'Authorization': f'Token {token}'}
            return base_url, header_gs
        else:
            print('로그인 실패')

def getTasksInfo():
    if debug:
        projectId = 1
    else:
        projectId = input('projectID: ')
    r = requests.get(base_url+f'projects/{projectId}', headers=header_gs)
    if r.ok:
        projectInfo = r.json()
        projectName = projectInfo['name']
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
        raise ConnectionError

    r = requests.get(base_url+f'projects/{projectId}/tasks', headers=header_gs)
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

        result = {'taskName': taskName,
                  'labels': labels,
                  'jobUrls': jobUrls,
                  'taskFrameCount': taskFrameCount,
                  'jobSize': jobSize,
                  'jobAssignee': jobAssignee,
                  'jobStatus': jobStatus,
                  'jobTaskId': jobTaskId,
                  'keys': keys}

        # Task name에서 type과 class 얻기
        result['jobLabelType'] = dict()
        result['jobLabelClass'] = dict()
        for jobId, _ in result['jobSize'].items():
            _, labelType, labelClass = result['taskName'][result['jobTaskId'][jobId]].split('_')
            result['jobLabelType'][jobId] = labelType
            result['jobLabelClass'][jobId] = labelClass
        return result
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
        raise ConnectionError

async def getAnnotationInfo(jobUrl):
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs)
    r = await loop.run_in_executor(None, get, jobUrl + "/annotations")
    if r.ok:
        return r
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def mkMonitoringFile():
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
    time = dt.datetime.now(gettz('Asia/Seoul')).strftime('%y%m%d-%H%M')
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
    labelerList = pd.read_csv("labeler_list.csv")
    for labeler in labelerList.itertuples():
        data = {
                "username": labeler.username,
                "email": labeler.email,
                "password1": labeler.password,
                "password2": labeler.password,
                "first_name": labeler.first_name,
                "last_name": labeler.last_name,
                }
        r = requests.post(base_url + "auth/register", data=data)

def assignLabeler(newAssignmentSize=1000):
    labelerListDf = pd.read_csv("labeler_list.csv")
    labelerList = labelerListDf['id']
    eachLabelerCR = getClassCR(labelerList, newAssignmentSize)
    assign(eachLabelerCR, labelerListDf)
    logWorkload(eachLabelerCR)


def assignment2Df(dict_:dict, classCRProj, newAssignmentSize=1000):
    seriesList = []
    for key, val in dict_.items():
        seriesList.append(pd.Series(val,name=key))

    classAsnDf = pd.DataFrame(seriesList).T.fillna(0)
    classAsnDf = classAsnDf.reindex(classCRProj.index, fill_value=0)
    classAsnDf['Rate'] = classAsnDf['Count']/classAsnDf['Count'].sum()
    newAssignmentTotal = newAssignmentSize-classAsnDf['annotation'].sum()
    classAsnDf['newAssignmentTarget'] = (classAsnDf['Count'].sum()+newAssignmentTotal)*classCRProj['Rate']-classAsnDf['Count']
    classAsnDf['newAssignmentActual'] = 0
    return {"newAssignmentSize":newAssignmentTotal, 'data':classAsnDf}

def getClassCR(labelerList, newAssignmentSize=1000):

    #project 전체의 이미지 수, 각 class의 비율
    classCountProjBbox = DefaultDict(int)
    classCountProjSeg = DefaultDict(int)
    for taskId, count in taskInfo['taskFrameCount'].items():
        _, labelType, labelClass = taskInfo['taskName'][taskId].split('_')
        if labelType=='Bbox':
            classCountProjBbox[f'{labelClass}'] += count
        elif labelType=='Bbox':
            classCountProjSeg[f'{labelClass}'] += count
        else:
            raise TypeError
    classCountProjBbox = pd.Series(classCountProjBbox)
    classRateProjBbox = classCountProjBbox/classCountProjBbox.sum()
    classCRProjBbox = pd.DataFrame([classCountProjBbox,classRateProjBbox], index=['Count', 'Rate']).T
    
    classCountProjSeg = pd.Series(classCountProjSeg)
    classRateProjSeg = classCountProjSeg/classCountProjSeg.sum()
    classCRProjSeg = pd.DataFrame([classCountProjSeg,classRateProjSeg], index=['Count', 'Rate']).T
    

    #assignee가 할당된 job의 이미지 수, 각 class의 비율
    classCountAsnlist = lambda : {'Bbox': {'Count':DefaultDict(int),
                                           'annotation':DefaultDict(int), 
                                           'validation':DefaultDict(int), 
                                           'modification':DefaultDict(int), 
                                           'completed':DefaultDict(int)},
                                  'Seg' : {'Count':DefaultDict(int),
                                           'annotation':DefaultDict(int), 
                                           'validation':DefaultDict(int), 
                                           'modification':DefaultDict(int), 
                                           'completed':DefaultDict(int)}}
    eachLabelerCR = DefaultDict(classCountAsnlist)
    for jobId, count in taskInfo['jobSize'].items():
        assigneeName = taskInfo['jobAssignee'][jobId]
        _, labelType, labelClass = taskInfo['taskName'][taskInfo['jobTaskId'][jobId]].split('_')
        if assigneeName != None:
            if labelType=='Bbox':
                eachLabelerCR[assigneeName]['Bbox']['Count'][labelClass] += count
                status =  taskInfo['jobStatus'][jobId]
                eachLabelerCR[assigneeName]['Bbox'][status][labelClass] += count
            elif labelType=='Seg':
                eachLabelerCR[assigneeName]['Seg'][labelClass] += count
                status =  taskInfo['jobStatus'][jobId]
                eachLabelerCR[assigneeName]['Seg'][status][labelClass] += count
            else:
                raise TypeError
    newEachLabelerCR = DefaultDict(dict)
    for assignee in labelerList:
        newEachLabelerCR[assignee]['Bbox'] = assignment2Df(eachLabelerCR[assignee]['Bbox'],classCRProjBbox, newAssignmentSize)
        newEachLabelerCR[assignee]['Seg'] = assignment2Df(eachLabelerCR[assignee]['Seg'], classCRProjSeg, newAssignmentSize)

    return newEachLabelerCR

def assign(eachLabelerCR, labelerListDF):
    labelerList = labelerListDF.loc[labelerListDF['activate'].notna(),'id']
    labelers = cycle(list(eachLabelerCR.items()))
    for jobId, assignee in taskInfo['jobAssignee'].items():
        if (assignee == None):
            loopStart = True
            while True:
                labeler, data = next(labelers)
                if not labelerList.isin(labeler):
                    continue
                if loopStart:
                    labeler0 = labeler
                    loopStart = False
                    continue
                if labeler0==labeler:
                    break
                if (data[taskInfo['jobLabelType'][jobId]]['newAssignmentSize']>0)&\
                    (data[taskInfo['jobLabelType'][jobId]]['data'].loc[taskInfo['jobLabelClass'][jobId],'newAssignmentTarget'] >= 0):
                    # requests.patch(base_url + f"jobs/{jobId}", headers=header_gs, data={"assignee": int(labeler)})
                    data[taskInfo['jobLabelType'][jobId]]['newAssignmentSize'] -= taskInfo['jobSize'][jobId]
                    data[taskInfo['jobLabelType'][jobId]]['data'].loc[taskInfo['jobLabelClass'][jobId],'newAssignmentTarget'] -= taskInfo['jobSize'][jobId]
                    data[taskInfo['jobLabelType'][jobId]]['data'].loc[taskInfo['jobLabelClass'][jobId],'newAssignmentActual'] += taskInfo['jobSize'][jobId]
                    taskInfo['jobAssignee'][jobId] = labeler
                    break

def logWorkload(eachLabelerCR):
    exportSum = []
    exportData = []
    sumIndexNames = ['time','assignee', 'labelType']
    classIndexNames = ['time','assignee', 'labelType', 'labelClass']
    cols = ['Count', 'annotation', 'validation', 'modification', 'completed']
    totalCols = [f'total {col}' for col in cols]
    gradCols = [f'grad {col}' for col in cols]
    execTime = dt.datetime.now(gettz('Asia/Seoul')).strftime('%y%m%d-%H%M')
    for assignee, labelType in eachLabelerCR.items():
        for type, data in labelType.items():
            classDf = data['data'].reindex(columns=cols)
            classDf.columns = totalCols
            sumDf = classDf.sum(axis=0)
            sumDf['assignee'] = classDf['assignee'] = assignee
            sumDf['labelType'] = classDf['labelType'] = type
            sumDf['time'] = classDf['time'] = execTime

            exportSum.append(sumDf)
            exportData.append(classDf.reset_index().rename(columns={'index':'labelClass'}))
    totalWorkloadClass = pd.concat(exportData).set_index(classIndexNames).reindex(columns=totalCols+gradCols)
    totalWorkloadSum = pd.DataFrame(exportSum).set_index(sumIndexNames).reindex(columns=totalCols+gradCols)


    # 기존 log 파일 여부 확인
    filename = 'workloadSum.csv'
    if os.path.isfile(filename):
        preWorkloadSum = pd.read_csv(filename)
        preWorkloadSum['time'] = preWorkloadSum['time'].map(dt.datetime.strftime('%y%m%d-%H%M'))
        preWorkloadSum = preWorkloadSum.set_index(sumIndexNames)
    else:
        preWorkloadSum = pd.DataFrame(columns=sumIndexNames+totalCols+gradCols).set_index(sumIndexNames)
        preWorkloadSum = preWorkloadSum.reindex(index=totalWorkloadSum.index, fill_value=0)
        

    recentTime = preWorkloadSum.index.get_level_values('time').max()
    recentWorkloadSum = preWorkloadSum.loc[preWorkloadSum.index.get_level_values('time')==recentTime,totalCols].copy(deep=True)
    recentWorkloadSum.index = totalWorkloadSum.index
    newWorkloadSum = totalWorkloadSum.reindex(columns=totalCols, fill_value=0)


    gradWorkloadSum = newWorkloadSum - recentWorkloadSum
    gradWorkloadSum.columns = gradCols
    newWorkloadSum = pd.concat([totalWorkloadSum, gradWorkloadSum], axis=1)

    if recentTime


    filename = 'workloadClass.csv'
    if os.path.isfile(filename):
        totalWorkloadClass = pd.read_csv(filename)
        totalWorkloadClass['time'] = totalWorkloadClass['time'].map(dt.datetime.strftime('%y%m%d-%H%M'))
        totalWorkloadClass = totalWorkloadClass.set_index(classIndexNames)
    else:
        totalClassDf.to_csv(filename)


            # preSumDf = workloadSum.reindex(index=[preTime, assignee, type])
            # preClassDf = workloadClass.reindex(index=[preTime, assignee, type])
            # gradDf = sumDf[totalCols] - preSumDf[totalCols]
            # sumDf['grad Count'] = classDf['grad Count'] = type
            # sumDf['grad annotation'] = classDf['grad annotation'] = type
            # sumDf['grad validation'] = classDf['grad validation'] = type
            # sumDf['grad modification'] = classDf['grad modification'] = type
            # sumDf['grad completed'] = classDf['grad completed'] = type


    print()



async def main():
    choice = None
    while choice != "0":
        print \
        ("""
        ---MENU---
        
        0 - Exit
        1 - Run
        2 - User Register
        3 - Assign Labeler
        """)

        choice = input("Your choice: ") # What To Do ???
        print()
    
        if choice == "0":
            print("Good bye!")  
        elif choice == "1":
            await mkMonitoringFile()
        elif choice == "2":
            registUser(base_url)
        elif choice == "3":
            assignLabeler( newAssignmentSize=1000)
        else:
            print(" ### Wrong option ### ")

### Main program    
if __name__ == "__main__":
    debug=True
    base_url, header_gs = login()
    taskInfo = getTasksInfo()
    aio.run(main())