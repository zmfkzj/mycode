from typing import DefaultDict
from dateutil.tz.tz import gettz
from numpy.lib.function_base import bartlett
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

#### FUNCTIONS ###
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

class AsyncList:
    def __init__(self, list):
        self.current = 0
        self.list = list
        self.len = len(list)
 
    def __aiter__(self):
        return self
 
    async def __anext__(self):
        if self.current < self.len:
            r = self.list[self.current]
            self.current += 1
            return r
        else:
            raise StopAsyncIteration

def getTasksInfo(base_url, header_gs):
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
        return result
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))
        raise ConnectionError

async def getAnnotationInfo(jobUrl, header_gs):
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs)
    r = await loop.run_in_executor(None, get, jobUrl + "/annotations")
    if r.ok:
        return r
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def mkMonitoringFile(base_url, header_gs):
    taskInfo = getTasksInfo(base_url, header_gs)
    aio_tasks = dict()
    datas = []
    async def getData(key):
        url = taskInfo['jobUrls'][key[3]]
        annotations = (await getAnnotationInfo(url,header_gs)).json()
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

    # dfCountObjectProjClass = df[['project','job_id','label_type','class']].groupby(['project','class','label_type']).count().unstack(level=-2)
    # dfCountObjectProjClass.columns = dfCountObjectProjClass.columns.droplevel(level=0)

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

def registUser(base_url):
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

def assignLabeler(base_url, header_gs, newAssignmentSize=1000):
    labelerListDf = pd.read_csv("labeler_list.csv")
    labelerList = labelerListDf.loc[labelerListDf['activate'].notna(), 'id']
    assignPolicy(base_url, header_gs,*getClassCR(base_url, header_gs, labelerList, newAssignmentSize))


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

def getClassCR(base_url, header_gs, labelerList, newAssignmentSize=1000):
    taskInfo = getTasksInfo(base_url, header_gs)

    #project 전체의 이미지 수, 각 class의 비율
    classCountProjBbox = DefaultDict(int)
    classCountProjSeg = DefaultDict(int)
    for taskId, count in taskInfo['taskFrameCount'].items():
        uploadDate, labelType, labelClass = taskInfo['taskName'][taskId].split('_')
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
    taskInfo['jobLabelType'] = dict()
    taskInfo['jobLabelClass'] = dict()
    for jobId, count in taskInfo['jobSize'].items():
        assigneeName = taskInfo['jobAssignee'][jobId]
        uploadDate, labelType, labelClass = taskInfo['taskName'][taskInfo['jobTaskId'][jobId]].split('_')
        taskInfo['jobLabelType'][jobId] = labelType
        taskInfo['jobLabelClass'][jobId] = labelClass
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

    return taskInfo, newEachLabelerCR

def assignPolicy(base_url, header_gs, taskInfo, eachLabelerCR):
    labelers = cycle(list(eachLabelerCR.items()))
    for jobId, assignee in taskInfo['jobAssignee'].items():
        if (assignee == None):
            loopStart = True
            while True:
                labeler, data = next(labelers)
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
    export = []
    for assignee, labelType in eachLabelerCR.items():
        for type, data in labelType.items():
            data['data']['assignee'] = assignee
            data['data']['labelType'] = type
            export.append(data['data'].reset_index())
    exportDf = pd.concat(export, ignore_index=True)


    return taskInfo


async def main():
    base_url, header_gs = login()
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
            await mkMonitoringFile(base_url, header_gs)
        elif choice == "2":
            registUser(base_url)
        elif choice == "3":
            assignLabeler(base_url, header_gs, newAssignmentSize=1000)
        else:
            print(" ### Wrong option ### ")

### Main program    
if __name__ == "__main__":
    debug=True
    aio.run(main())