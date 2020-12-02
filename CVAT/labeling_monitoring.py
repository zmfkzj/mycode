from dateutil.tz.tz import gettz
import pandas as pd
import asyncio as aio
import requests
from functools import partial
from getpass import getpass
from tqdm.asyncio import tqdm
from copy import deepcopy
import datetime as dt
import math
from easydict import EasyDict
import json

def readJson():
    with open('config_NH.json','r') as f:
        cfg = json.load(f)
    return cfg

def getBaseUrl():
    if cfg.base_url.endswith('/'):
        base_url = '{}api/v1/'.format(cfg['base_url'])
    else:
        base_url = '{}/api/v1/'.format(cfg['base_url'])
    return base_url

#### FUNCTIONS ###
def login():
    while True:
        print("Getting token...")
        api_login,api_password = auth()
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

async def getTasksInfo(base_url, header_gs):
    r = requests.get(base_url + "tasks", headers=header_gs, params={'page_size':1})
    r = requests.get(base_url + "tasks", headers=header_gs, params={'page_size':r.json()['count']})
    if r.ok:
        taskName = dict()
        labels = dict()
        jobUrls = dict()
        jobSize = dict()
        jobAssignee = dict()
        jobStatus = dict()
        keys = []
        taskFrameCount = dict()

        async def getInfo(task):
            meta = await getTaskMeta(task['url'], header_gs)
            taskFrameCount[task['id']] = meta['size']

            key = [task['project'],task['id'], task['name']]
            taskName[task['id']] = task['name']
            for label in task['labels']:
                labels[label['id']] = label['name']
            for job in task['segments']:
                jobSize[job['jobs'][0]['id']] = job['stop_frame']-job['start_frame']+1
                jobUrls[job['jobs'][0]['id']] = job['jobs'][0]['url']
                jobAssignee[job['jobs'][0]['id']] = job['jobs'][0]['assignee']
                jobStatus[job['jobs'][0]['id']] = job['jobs'][0]['status']
                _key = key+[job['jobs'][0]['id']]
                keys.append(_key)

        aio_task = []
        for task in r.json()['results']:
            aio_task.append(aio.create_task(getInfo(task)))
        await aio.gather(*aio_task)
        # async for task in tqdm(AsyncList(r.json()['results']), desc='Task Info'):
        result = {'taskName': taskName,
                  'labels': labels,
                  'jobUrls': jobUrls,
                  'taskFrameCount': taskFrameCount,
                  'jobSize': jobSize,
                  'jobAssignee': jobAssignee,
                  'jobStatus': jobStatus,
                  'keys': keys}
        return result
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def getAnnotationInfo(jobUrl, header_gs):
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs)
    r = await loop.run_in_executor(None, get, jobUrl + "/annotations")
    if r.ok:
        return r
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def getTaskMeta(taskUrl, header_gs):
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs)
    r = await loop.run_in_executor(None, get, taskUrl + "/data/meta")
    if r.ok:
        return r.json()
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def mkMonitoringFile(base_url, header_gs):
    taskInfo = await getTasksInfo(base_url, header_gs)
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
    else:
        api_login = 'serveradmin'
        print(api_login)
        api_password = 'wnrWkd131@Cv'
    return api_login, api_password

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

def assignLabeler(base_url, header_gs):
    assignTable = pd.read_csv('assign.csv')
    for assign in assignTable.itertuples():
        if not math.isnan(assign.assignee):
            r = requests.patch(base_url + f"jobs/{assign.job_id}", headers=header_gs, data={"assignee": int(assign.assignee)})
            # r = requests.patch(base_url + f"jobs/{assign.job_id}", data={"assignee":assign.assignee})

async def main():
    base_url, header_gs = login()
    choice = None
    while choice != "0":
        print \
        ("""
        ---MENU---
        
        0 - Exit
        1 - Run
        """)

        choice = input("Your choice: ") # What To Do ???
        print()
    
        if choice == "0":
            print("Good bye!")  
        elif choice == "1":
            await mkMonitoringFile(base_url, header_gs)
        else:
            print(" ### Wrong option ### ")

### Main program    
if __name__ == "__main__":
    debug=False
    cfg = EasyDict(readJson())
    base_url = getBaseUrl()
    aio.run(main())