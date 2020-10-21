from dateutil.tz.tz import gettz
import pandas as pd
import asyncio as aio
import requests
from functools import partial
from getpass import getpass
from tqdm.asyncio import tqdm
from copy import deepcopy
import datetime as dt

#### FUNCTIONS ###
def login(base_url,api_login,api_password):
    print("Getting token...")
    data_get = {'username': api_login,
                'password': api_password}
    r = requests.post(base_url + 'auth/login', data=data_get)
    if r.ok:
        cookies = dict(r.cookies)
        return cookies
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

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

async def getTasksInfo(base_url, cookies):
    print("Checking session...")
    header_gs = {'Accept': 'application/json'}
    r = requests.get(base_url + "tasks", headers=header_gs, cookies=cookies, params={'page_size':1})
    r = requests.get(base_url + "tasks", headers=header_gs, cookies=cookies, params={'page_size':r.json()['count']})
    if r.ok:
        taskName = dict()
        labels = dict()
        jobUrls = dict()
        keys = []
        taskFrameCount = dict()

        async def getInfo(task):
            meta = await getTaskMeta(task['url'], cookies)
            taskFrameCount[task['id']] = meta['size']

            key = [task['project'],task['id'], task['name']]
            taskName[task['id']] = task['name']
            for label in task['labels']:
                labels[label['id']] = label['name']
            for job in task['segments']:
                jobUrls[job['jobs'][0]['id']] = job['jobs'][0]['url']
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
                  'keys': keys}
        return result
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def getAnnotationInfo(jobUrl, cookies):
    header_gs = {'Accept': 'application/json'}
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs, cookies=cookies)
    r = await loop.run_in_executor(None, get, jobUrl + "/annotations")
    if r.ok:
        return r
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def getTaskMeta(taskUrl, cookies):
    header_gs = {'Accept': 'application/json'}
    loop = aio.get_event_loop()
    get = partial(requests.get, headers=header_gs, cookies=cookies)
    r = await loop.run_in_executor(None, get, taskUrl + "/data/meta")
    if r.ok:
        return r.json()
    else:
        print("HTTP %i - %s, Message %s" % (r.status_code, r.reason, r.text))

async def main():
    # api_login = input('id: ')
    # api_password = getpass('pass: ')
    # port = input('port: ')

    api_login = 'serveradmin'
    api_password = 'wnrWkd131@Cv'
    port = '9100'
    base_url = "http://tmlabel.asuscomm.com:9100/api/v1/"

    cookies = login(base_url,api_login,api_password)
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
            # taskName, labels, jobUrls, keys = getTasksInfo(base_url, cookies)
            taskInfo = await getTasksInfo(base_url, cookies)
            aio_tasks = dict()
            datas = []
            async def asdf(key):
                url = taskInfo['jobUrls'][key[3]]
                annotations = (await getAnnotationInfo(url,cookies)).json()
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
                aio_tasks.append(aio.create_task(asdf(key)))

            async for task in tqdm(aio_tasks):
                await task

            cols = ['project','task_id','task_name','job_id','label_type','frame','class']
            df = pd.DataFrame(datas, columns=cols)
            df['project'] = df['project'].fillna('default')
            dfTaskClass = df[['task_id','task_name','job_id','label_type','class']].groupby(['task_id','task_name','class','label_type']).count().unstack(-2)
            dfProjClass = df[['project','job_id','label_type','class']].groupby(['project','class','label_type']).count().unstack(level=-2)
            dfTaskClass.columns = dfTaskClass.columns.droplevel(level=0)
            dfProjClass.columns = dfProjClass.columns.droplevel(level=0)

            dfTaskImg = df[['project','task_id','task_name','frame']].drop_duplicates(['project','task_id','frame']).groupby(['project','task_id','task_name']).count()
            dfTaskImg.columns = ['isin']
            dfTaskImg['image_size'] = dfTaskImg.index.droplevel(level=[0,-1]).map(taskInfo['taskFrameCount'])
            dfTaskImg['not isin'] = dfTaskImg['image_size']-dfTaskImg['isin']

            dfProjImg = dfTaskImg.groupby(['project']).sum()

            time = dt.datetime.now(gettz('Asia/Seoul')).strftime('%y%m%d-%H%M')
            dfTaskClass.to_csv(f'./{time}_CVAT-Class-Task.csv', encoding='euc-kr')
            dfProjClass.to_csv(f'./{time}_CVAT-Class-Proj.csv', encoding='euc-kr')
            dfTaskImg.to_csv(f'./{time}_CVAT-image-Task.csv', encoding='euc-kr')
            dfProjImg.to_csv(f'./{time}_CVAT-image-Proj.csv', encoding='euc-kr')
        else:
            print(" ### Wrong option ### ")

### Main program    
aio.run(main())