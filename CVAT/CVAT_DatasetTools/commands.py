from collections import defaultdict
from pathlib import Path
import sys
import os
import stat
from easydict import EasyDict
from typing import Callable, OrderedDict,Dict
import random as rd
from datumaro.components.project import Project # project-related things
from datumaro.components.extractor import DatasetItem, Bbox
from datumaro.cli.__main__ import main
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import asyncio as aio
import numpy as np

osName = os.name

def consolClear():
    if osName == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

class Menu:
    prevMunu = None

    def __init__(self, menuName:str) -> None:
        self.menuName = menuName
        self.command = EasyDict()
        if menuName != 'Main':
            self.setCommand('이전메뉴', self.goToPrev)
        self.setCommand('종료', sys.exit)

    def __str__(self) -> str:
        header = 'No\tOption\t\t\tHelp\n-----------------------------------------------------------------'
        opts = [self.menuName, header]
        for no,val in self.command.items():
            line = f'{no}\t{val.opt}\t\t{val.help}'
            opts.append(line)
        return '\n'.join(opts)
    
    def __call__(self):
        self.selectAndRun()

    def selectAndRun(self):
        self.prevMunu = Menu.prevMunu
        Menu.prevMunu = self
        consolClear()
        print(self)
        while True:
            selNum = input('선택 No: ').strip()
            if selNum in self.command.keys():
                break
            else:
                print('입력이 잘못되었습니다. ')
        self.command[selNum].function()
        self.selectAndRun()

    def setCommand(self, opt:str, function:Callable, help='') :
        no = str(len(self.command))
        self.command[no] = {"opt": opt,
                            "function": function, 
                            "help": help}
        return self

    def goToPrev(self):
        self.prevMunu.selectAndRun()

class Arg:
    def __init__(self, argsAndType:Dict[str,str]=dict()):
        self.args = EasyDict()
        for name,(type, help) in argsAndType.items():
            inputPrint = f'\n\n{help}\n{name} 입력: '
            val = input(inputPrint)
            arg = eval(f'{type}(\'{val.strip()}\')')
            self.args[name] = arg
    
    def __getitem__(self, name) -> EasyDict: 
        return self.args[name]

class ImportArg(Arg):
    def __init__(self) -> None:
        self.supportFormat = ['coco', 'cvat', 'datumaro', 'image_dir', 'imagenet', 'imagenet_txt', 'label_me', 'mot_seq', 'mots','tf_detection_api', 'voc', 'yolo']
        args = {'format':('str', f'다운로드 한 데이터셋의 형식.\n지원형식: {",".join(self.supportFormat)}')}
        super().__init__(args)

class ExportArg(Arg):
    def __init__(self) -> None:
        self.supportFormat = ['coco', 'cvat', 'datumaro', 'datumaro_project', 'label_me', 'mot_seq_gt', 'mots_png', 'tf_detection_api', 'voc','voc_segmentation', 'yolo']
        args = {'format':('str', f'내보낼 데이터셋의 형식.\n지원형식: {",".join(self.supportFormat)}')}
        super().__init__(args)

class DrawUtil:
    colorMap = defaultdict(lambda :(rd.randint(0,256),rd.randint(0,256),rd.randint(0,256)))

    def __init__(self, root:Path, item:DatasetItem, colorMap:dict=None) -> None:
        self.item = item
        self.root = root
        self.imgPath = root/'images'/self.item.image
        self.loop = aio.get_event_loop()
        self.lineStyles = ['dot', 'solid']
        if colorMap is not None:
            DrawUtil.colorMap.update(colorMap)

    async def readImg(self):
        img = await self.loop.run_in_executor(None, Image.open, self.imgPath)
        self.img = np.array(img)
        self.fontscale = max(img.shape[:2])*(10/250)
        self.thick = int(max(img.shape[:2])*(1/250))
        if self.thick <= 0: self.thick=1
        return self

    async def saveImg(self):
        img = Image.fromarray(self.img)
        savePath:Path = self.root/'images_draw-object'/self.item.image
        savePath.mkdir(exist_ok=True, parents=True)
        await self.loop.run_in_executor(None, img.save, savePath)

    def roundRectangle(self, bbox:Bbox, linestyle='solid'):
        if linestyle=='solid':
            drline = cv2.line
            drellipsis = cv2.ellipse
        elif linestyle=='dot':
            drline = self.dotline
            drellipsis = self.dotellipse
        else:
            raise AssertionError(f'linestyle must be one of {", ".join(self.lineStyles)}')

        category = bbox.label
        b_h = int(bbox.h)
        b_w = int(bbox.w)
        topleft = tuple([int(i) for i in bbox.points[:2]])
        bottomright = tuple([int(i) for i in bbox.points[2:]])

        border_radius = self.thick*20
        r_y = min(border_radius,b_h)
        r_x = min(border_radius,b_w)

        drline(self.img, topleft, (bottomright[0]-r_x, topleft[1]), DrawUtil.colorMap[category], self.thick)#top
        drline(self.img, (topleft[0]+r_x,bottomright[1]), (bottomright[0]-r_x, bottomright[1]), DrawUtil.colorMap[category], self.thick)#bottom
        drline(self.img, topleft, (topleft[0], bottomright[1]-r_y), DrawUtil.colorMap[category], self.thick)#left
        drline(self.img, (bottomright[0],topleft[1]+r_y), (bottomright[0],bottomright[1]-r_y), DrawUtil.colorMap[category], self.thick)#right
        drellipsis(self.img, (bottomright[0]-r_x, topleft[1]+r_y), (r_x, r_y), 0, 0, -90, DrawUtil.colorMap[category], self.thick)#top-right
        drellipsis(self.img, (topleft[0]+r_x, bottomright[1]-r_y), (r_x, r_y), 0, 90, 180, DrawUtil.colorMap[category], self.thick)#bottom-left
        drellipsis(self.img, (bottomright[0]-r_x, bottomright[1]-r_y), (r_x, r_y), 0, 0, 90, DrawUtil.colorMap[category], self.thick)#bottom-right

    def dotellipse(self, center, r, rotation, start, end, color, thick):
        dr = int((end-start)/4.5)

        start1 = start
        while np.sign(end-start1)==np.sign(dr):
            end1 = start1+dr
            if np.abs(end-start1)< np.abs(dr):
                end1=end
            cv2.ellipse(self.img, center, r, rotation, start1, end1, color, thick)
            start1 += 2*dr

    def dotline(img, topleft, bottomright, color, thick):
        a = np.sqrt((bottomright[0]-topleft[0])**2+(bottomright[1]-topleft[1])**2)
        if a==0:
            return
        dotgap = thick*10
        b = a/dotgap
        dx = int((bottomright[0]-topleft[0])/b)
        dy = int((bottomright[1]-topleft[1])/b)

        x1, y1 = topleft
        while (np.sign(bottomright[0]-x1)==np.sign(dx)) & (np.sign(bottomright[1]-y1)==np.sign(dy)):
            end_x = x1+dx
            end_y = y1+dy

            if np.abs(bottomright[0]-end_x)<np.abs(dx):
                end_x = bottomright[0]
            if np.abs(bottomright[1]-end_y)<np.abs(dy):
                end_y = bottomright[1]
                
            cv2.line(img, (x1, y1), (end_x, end_y), color, thick)
            x1 += 2*dx
            y1 += 2*dy
    def drawbbs(self, lineStyle, cornerStyle, thick=1, color=(255,0,0)):
        '''
        bbox = (left, top, right, bottom)
        '''
        assert gt in [False, 'dot', 'solid'], 'gt must be False:bool,\'dot\' or \'solid\''
        color = color[::-1]
        bbox = list(map(lambda coord: int(np.around(coord)),bbox))
        if gt:
            RoundRectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick, linestyle=linestyle)
        else:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

        return img

    def drawlabel(img, label, bbox, fontscale=1, thick=1, color=(255,0,0)):
        color = color[::-1]
        bbox = list(map(lambda coord: int(np.around(coord)),bbox))
        #draw label
        osname = platform.system()
        if osname == 'Windows':
            try:
                fontpath = 'C:\\Users\\qlwla\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumGothicBold.ttf'
            except:
                fontpath = 'C:\\Windows\\Fonts\\malgunbd.ttf'
        else:
            fontpath = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"     
        font = ImageFont.truetype(fontpath, int(fontscale))
        img_label = Image.new('RGB', (int(fontscale*100),int(fontscale*1.5)))
        draw = ImageDraw.Draw(img_label)
        w, h = draw.textsize(label, font=font)

        draw.rectangle((0, 0, w-1, int(h*1.1)), fill=color, width=-1)
        draw.rectangle((0, 0, w-1, int(h*1.1)), fill=color, width=thick)
        draw.text((0, 0), label, font = font, fill = (1,1,1))
        img_label = np.array(img_label)

        text_y = bbox[1] - h #label 위치 조정
        if text_y<0:
            text_y=0
        img_shape = img.shape
        label_idx = np.where(img_label>0)
        label_idx_inImg = [label_idx[0]+text_y, label_idx[1]+bbox[0], label_idx[2]]
        for idx, val in enumerate(label_idx_inImg[:2]):
            out_range = np.where(val>=img_shape[idx])
            label_idx = [np.delete(i, out_range) for i in label_idx]
            label_idx_inImg = [np.delete(i, out_range) for i in label_idx_inImg]
        img[tuple(label_idx_inImg)] = img_label[tuple(label_idx)]

        return img




class Commands:
    def __init__(self) -> None:
        root = Tk()
        self.datasetsPath = Path(filedialog.askdirectory())
        root.destroy()
        self.projectsPath = (Path('projects')).absolute()
        self.mergeFolderName = 'merged'

    def checkDefineVariable(self):
        try:
            self.projectsPathListFromDataset
            return True
        except AttributeError:
            return False

    def importDataset(self, args:Arg):
        if args['format'].lower() not in args.supportFormat:
            args['format'] = input('지원하지 않는 format입니다. 다시 입력해주세요.')
        self.projectsPath.mkdir(exist_ok=True, parents=True)
        self.projectsPathListFromDataset = [self.projectsPath/path.name for path in self.getSubDirList(self.datasetsPath)]
        datasetPathList = [path for path in self.getSubDirList(self.datasetsPath) if path.is_dir()]
        for datasetPath in datasetPathList:
            projPath = self.projectsPath/datasetPath.name
            projImportArgs = ['project', 'import', '-i', str(datasetPath), '-o', str(projPath), '-f', args['format'].lower(), '--overwrite']
            main(projImportArgs)
        return self

    def mergeDataset(self):
        if not self.checkDefineVariable():
            importArgs = ImportArg()
            self.importDataset(importArgs)
        projsPathList = [str(dir) for dir in self.projectsPathListFromDataset]
        mergePath = (self.projectsPath/self.mergeFolderName)
        mergePath.mkdir(exist_ok=True)
        merge_args = ['merge', '-o', str(mergePath), '--overwrite', *projsPathList]
        main(merge_args)
        return self

    def exportDataset(self, args:Arg, merge=False):
        if args['format'].lower() not in args.supportFormat:
            args['format'] = input('지원하지 않는 format입니다. 다시 입력해주세요.')

        if merge:
            projectsPathList = [self.projectsPath/self.mergeFolderName]
        else:
            if not self.checkDefineVariable():
                importArgs = ImportArg()
                self.importDataset(importArgs)
            projectsPathList = self.projectsPathListFromDataset

        for proj in projectsPathList:
            exportPath = (Path('exportDataset')/args['format'].lower()/proj.name).absolute()
            exportPath.mkdir(exist_ok=True, parents=True)
            export_args = ['project','export','-f',args['format'].lower(),'-o',str(exportPath),'-p',str(proj)]
            main(export_args)
        return self


    def getSubDirList(self,Path:Path):
        return [dir for dir in Path.iterdir() if dir.is_dir()]

    def mergeFunction(self):
        importArgs = ImportArg()
        exportArgs = ExportArg()
        self.importDataset(importArgs)
        self.mergeDataset()
        self.exportDataset(exportArgs, merge=True)
        return self

    def convertFunction(self):
        importArgs = ImportArg()
        exportArgs = ExportArg()
        self.importDataset(importArgs)
        self.exportDataset(exportArgs)
        return self

    def loadDatasetFromProjFolder(self):
        if not self.checkDefineVariable():
            importArgs = ImportArg()
            self.importDataset(importArgs)
        projectFolders = [path for path in self.projectsPathListFromDataset]
        projects = [Project.load(projectFolder) for projectFolder in projectFolders]
        self.datasets = [project.make_dataset() for project in projects]
        return self





if __name__ == "__main__":
    command = Commands()
    mainMenu = Menu('Main').setCommand('dataset 합치기',command.mergeFunction,'여러 데이터셋을 하나로 합칩니다.')\
                        .setCommand('dataset 변환', command.convertFunction, '다른 형식의 데이터셋으로 바꿉니다. 예) coco format -> voc format')\
                        .setCommand('loadProjects', command.loadDatasetFromProjFolder)

    mainMenu()