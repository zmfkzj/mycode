from collections import defaultdict
from pathlib import Path
import sys
import os
import stat
from easydict import EasyDict
from typing import Callable, OrderedDict,Dict, Union
import random as rd
from datumaro.components.project import Project, ProjectDataset # project-related things
from datumaro.components.extractor import DatasetItem, Bbox, Polygon, AnnotationType, LabelCategories
from datumaro.cli.__main__ import main
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont
import cv2
import asyncio as aio
import numpy as np
from tqdm import tqdm
from tqdm import asyncio


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

class DrawItemArg(Arg):
    def __init__(self) -> None:
        args = {'lineStyle':('str', f'선의 형태. dot(점선), solid(실선)'),
                'cornerStyle':('str', f'상자 모서리의 형태. sharp(직각), round(둥금)')}
        super().__init__(args)

class customDataset:

    def __init__(self, dataset:ProjectDataset) -> None:
        self.dataset = dataset
        categories = dataset.categories()[AnnotationType.label]
        self._imageDatas = (customDataset.ImageData(item, categories) for item in dataset)

    def drawAndExport(self, lineStyle, cornerStyle):
        for imageData in tqdm(self._imageDatas, total=len(self.dataset)):
            imageData.drawItem(lineStyle, cornerStyle).saveImg()

    class ImageData:
        colorMap = defaultdict(lambda :(rd.randint(0,64)*4+3,rd.randint(0,64)*4+3,rd.randint(0,64)*4+3))
        try:
            with open(str(self.root/'labelmap.txt'),'r', encoding='utf-8') as f:
                labelmap = f.readlines()
            labelmap = [line.split(':')[0] for line in labelmap]
            labelmap = dict([(line[0], line[1]) for line in labelmap])
            colorMap.update(labelmap)
            print(colorMap)
        except:
            print('label color를 임의로 생성합니다.')

        def __init__(self, item:DatasetItem, categories:LabelCategories) -> None:
            self.item = item
            self.lineStyles = ['dot', 'solid']
            self.conerStyles = ['sharp', 'round']
            self.categories = categories
            self.img = item.image.data
            self.fontscale = max(self.img.shape[:2]*np.array([30/1080, 30/1620]))
            self.thick = int(max([*list(self.img.shape[:2]*np.array([2/1080, 2/1620])),2]))
            self.root = Path(item.image.path[:item.image.path.rfind(item.id)]).parent

        # async def readImg(self):
        #     img = await customDataset.loop.run_in_executor(None, Image.open, self.item.path)
        #     self.img = np.array(img)
        #     self.fontscale = max(img.shape[:2]*np.array([10/1080, 10/1620]))
        #     self.thick = int(max(img.shape[:2]*np.array([1/1080, 1/1620])))
        #     return self

        def saveImg(self):
            img = Image.fromarray(self.img.astype(np.uint8))
            savePath:Path = self.root/'images_draw-label'/Path(self.item.image.path).name
            savePath.parent.mkdir(exist_ok=True, parents=True)
            img.save(savePath)
            return self

        def drawItem(self, lineStyle, cornerStyle):
            for anno in self.item.annotations:
                if isinstance(anno,Bbox):
                    self.drawBbox(anno, lineStyle, cornerStyle).drawLabel(anno)
                elif isinstance(anno,Polygon):
                    self.drawSeg()
            return self

        @staticmethod
        def getColor(anno:Union[Bbox,Polygon]):
            color = customDataset.ImageData.colorMap[anno.label]
            while len(customDataset.ImageData.colorMap) != len(set(customDataset.ImageData.colorMap.values())):
                del customDataset.ImageData.colorMap[anno.label]
                color = customDataset.ImageData.colorMap[anno.label]
            return color

        def drawBbox(self, anno:Bbox, lineStyle, cornerStyle):
            color = self.getColor(anno)
            bbox = [int(i) for i in anno.points]
            if cornerStyle=='round':
                self.roundRectangle(self.img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color, self.thick, linestyle=lineStyle)
            elif cornerStyle=='sharp':
                self.rectangle(self.img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, self.thick, linestyle=lineStyle)
            else:
                raise AssertionError(f'cornerStyle must be one of {", ".join(self.conerStyles)}')
            return self

        def rectangle(self, img, topleft, bottomright, color, thick, linestyle='solid'):
            if linestyle=='solid':
                cv2.rectangle(img, (topleft[0], topleft[1]), (bottomright[0], bottomright[1]), color, thick)
            elif linestyle=='dot':
                self.dotLine(img, topleft, (bottomright[0], topleft[1]), color, thick)#top
                self.dotLine(img, (topleft[0],bottomright[1]), (bottomright[0], bottomright[1]), color, thick)#bottom
                self.dotLine(img, topleft, (topleft[0], bottomright[1]), color, thick)#left
                self.dotLine(img, (bottomright[0],topleft[1]), (bottomright[0],bottomright[1]), color, thick)#right
            return self
            
        def roundRectangle(self, img, topleft, bottomright, color, thick, linestyle='solid'):
            if linestyle=='solid':
                _line, _ellipsis = cv2.line, cv2.ellipse
            elif linestyle=='dot':
                _line, _ellipsis = self.dotLine, self.dotEllipse
            else:
                raise AssertionError(f'linestyle must be one of {", ".join(self.lineStyles)}')

            border_radius = thick*20
            b_h, b_w = int((bottomright[1]-topleft[1])/2), int((bottomright[0]-topleft[0])/2)
            r_y, r_x = min(border_radius,b_h), min(border_radius,b_w)

            _line(img, topleft, (bottomright[0]-r_x, topleft[1]), color, thick)#top
            _line(img, (topleft[0]+r_x,bottomright[1]), (bottomright[0]-r_x, bottomright[1]), color, thick)#bottom
            _line(img, topleft, (topleft[0], bottomright[1]-r_y), color, thick)#left
            _line(img, (bottomright[0],topleft[1]+r_y), (bottomright[0],bottomright[1]-r_y), color, thick)#right
            _ellipsis(img, (bottomright[0]-r_x, topleft[1]+r_y), (r_x, r_y), 0, 0, -90, color, thick)#top-right
            _ellipsis(img, (topleft[0]+r_x, bottomright[1]-r_y), (r_x, r_y), 0, 90, 180, color, thick)#bottom-left
            _ellipsis(img, (bottomright[0]-r_x, bottomright[1]-r_y), (r_x, r_y), 0, 0, 90, color, thick)#bottom-right
            return self

        @staticmethod
        def dotEllipse(img, center, r, rotation, start, end, color, thick):
            dr = int((end-start)/4.5)

            start1 = start
            while np.sign(end-start1)==np.sign(dr):
                end1 = start1+dr
                if np.abs(end-start1)< np.abs(dr):
                    end1=end
                cv2.ellipse(img, center, r, rotation, start1, end1, color, thick)
                start1 += 2*dr

        @staticmethod
        def dotLine(img, topleft, bottomright, color, thick):
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

        def drawLabel(self, anno:Union[Bbox,Polygon]):
            bbox = list(map(lambda coord: int(np.around(coord)),anno.points))
            label = self.categories[anno.label].name
            color = self.getColor(anno)
            textColor = tuple(np.array([255,255,255]) - np.array(color))
            #draw label
            fontpath = 'NanumGothicBold.ttf'
            font = ImageFont.truetype(fontpath, int(self.fontscale))
            img_label = Image.new('RGB', (int(self.fontscale*100),int(self.fontscale*1.5)),color=color)
            draw = ImageDraw.Draw(img_label)
            draw.text((0, 0), label, font = font, fill = textColor)
            w, h = draw.textsize(label, font=font)
            img_label = img_label.crop((0,0,w,int(h*1.1)))
            text_y = max(bbox[1] - h,0) #label 위치 조정
            img = Image.fromarray(self.img.astype(np.uint8))
            img.paste(img_label,(bbox[0],text_y))
            self.img = np.array(img)

            # label_idx_inImg = (bbox[0], text_y, bbox[0]+w, text_y+h)
            # self.img[label_idx_inImg[1]:label_idx_inImg[3], label_idx_inImg[0]:label_idx_inImg[2]] = img_label
            return self

        def drawSeg(self):
            return

class Commands:
    def __init__(self) -> None:
        root = Tk()
        self.datasetsPath = Path(filedialog.askdirectory())
        root.destroy()
        self.projectsPath = (self.datasetsPath/'..'/'projects').resolve().absolute()
        self.mergeFolderName = 'merged'

    def checkDefineVariable(self, var):
        try:
            eval(var)
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
            if not self.checkDefineVariable('self.projectsPathListFromDataset'):
                importArgs = ImportArg()
                self.importDataset(importArgs)
            projectsPathList = self.projectsPathListFromDataset

        for proj in projectsPathList:
            exportPath = (Path('exportDataset')/args['format'].lower()/proj.name).absolute()
            exportPath.mkdir(exist_ok=True, parents=True)
            export_args = ['project','export','-f',args['format'].lower(),'-o',str(exportPath),'-p',str(proj)]
            main(export_args)
        return self


    @staticmethod
    def getSubDirList(Path:Path):
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

    def drawItemFuncion(self):
        importArgs = ImportArg()
        drawItemArgs = DrawItemArg()
        self.importDataset(importArgs)
        self.loadDatasetFromProjFolder()
        self.drawItemOfEveryDataset(drawItemArgs)
        return self

    def loadDatasetFromProjFolder(self):
        if not self.checkDefineVariable('self.projectsPathListFromDataset'):
            importArgs = ImportArg()
            self.importDataset(importArgs)
        projectFolders = [path for path in self.projectsPathListFromDataset]
        projects = [Project.load(projectFolder) for projectFolder in projectFolders]
        self.datasets = [customDataset(project.make_dataset()) for project in projects]
        return self
    
    def drawItemOfEveryDataset(self, args:Arg):
        if not self.checkDefineVariable('self.datasets'):
            self.loadDatasetFromProjFolder()
        else:
            pass
        for dataset in tqdm(self.datasets):
            dataset.drawAndExport(args['lineStyle'], args['cornerStyle'])
        return self

if __name__ == "__main__":
    command = Commands()
    mainMenu = Menu('Main').setCommand('dataset 합치기',command.mergeFunction,'여러 데이터셋을 하나로 합칩니다.')\
                        .setCommand('dataset 변환', command.convertFunction, '다른 형식의 데이터셋으로 바꿉니다. 예) coco format -> voc format')\
                        .setCommand('draw label', command.drawItemFuncion, '이미지에 라벨을 그리기. 각 데이터셋 폴더/images_draw-label 안에 저장됨.')

    mainMenu()