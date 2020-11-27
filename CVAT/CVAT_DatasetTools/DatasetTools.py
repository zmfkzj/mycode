from pathlib import Path
file__ = Path(__file__).parent
import sys
sys.path.insert(0,str(file__/'datumaro'))

# from datumaro.components.project import Project
from datumaro.cli.__main__ import main
import os
from easydict import EasyDict
from typing import Callable, OrderedDict
from shutil import rmtree

osName = os.name

def consolClear():
    if osName == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class Menu:
    prevMunu = None

    def __init__(self, menuName:str) -> None:
        self.menuName = menuName
        self.command = EasyDict()
        self.setCommand('이전메뉴', self.goToPrev)

    def __str__(self) -> str:
        header = 'No\tOption\t\tHelp\n------------------------------------'
        opts = [self.menuName, header]
        for no,val in self.command.items():
            line = f'{no}\t\t{val.opt}\t{val.help}'
            opts.append(line)
        return '\n'.join(opts)
    
    def __call__(self):
        self.selectAndRun()

    def selectAndRun(self):
        self.prevMunu = Menu.prevMunu
        Menu.prevMunu = self
        consolClear()
        print(self)
        selNum = int(input('선택 No: ').strip())
        self.command[selNum].function(*self.command.argValues)

    def setCommand(self, opt:str, function:Callable, help='') :
        self.command[len(self.command)] = {"opt": opt,
                                           "function": function, 
                                           "help": help}
        return self

    def goToPrev(self):
        self.prevMunu.selectAndRun()

class CLI:
    def __init__(self) -> None:
        self.datasetsPath = Path('datasets').absolute().mkdir(exist_ok=True)
        self.projectsPath = (Path('tmp/projects')).absolute().mkdir(exist_ok=True)

    def importDataset(self):
        args = self.setArg({'format':('str', '다운로드 한 데이터셋의 형식.\n지원형식: coco, cvat, datumaro, image_dir, imagenet, imagenet_txt, label_me, mot_seq, mots,tf_detection_api, voc, yolo')})
        datasetPathList = self.getSubDirList(self.datasetsPath)
        for datasetPath in datasetPathList:
            projPath = str(self.projectsPath/datasetPath.name)
            projImportArgs = ['project', 'import', '-i', datasetPath, '-o', projPath, '-f', args.format, '--overwrite']
            main(projImportArgs)
        return self

    def mergeDataset(self):
        projsPathList = self.getSubDirList(self.projectsPath)
        mergePath = (self.projectsPath/'mergedProject').mkdir(exist_ok=True)
        merge_args = ['merge', '-o', mergePath, '--overwrite', *projsPathList]
        main(merge_args)
        return self

    def exportDataset(self):
        args = self.setArg({'format':('str', '내보낼 데이터셋의 형식.\n지원형식:coco, cvat, datumaro, datumaro_project, label_me, mot_seq_gt, mots_png, tf_detection_api, voc,voc_segmentation, yolo')})
        projectsPathList = self.getSubDirList(self.projectsPath)
        for proj in projectsPathList:
            exportPath = (Path('exportDataset')/args.format/proj.name).absolute().mkdir(exist_ok=True)
            export_args = ['project','export','-f',args.format,'-o',str(exportPath),'-p',str(proj)]
            main(export_args)
        rmtree(self.projectsPath, ignore_errors=True)
        return self

    def setArg(self, argsAndType:dict[str,str]=dict())->EasyDict:
        args = EasyDict()
        for name,type in argsAndType.items():
            inputPrint = f'{help}\n\n{name}:{type} 입력: '
            val = input(inputPrint)
            arg = eval(f'{type}({val.strip()})')
            args[name] = arg
        return args

    def getSubDirList(self,Path:Path):
        return [dir for dir in Path.iterdir() if dir.is_dir()]

    def mergeFunction(self):
        self.importDataset()
        self.mergeDataset()
        self.exportDataset()
        return self

    def convertFunction(self):
        self.importDataset()
        self.exportDataset()
        return self

    #     self.projs = []
    #     for dir in self.datasetPath.iterdir():
    #         if dir.is_dir():
    #             datasetPath = str(dir.absolute())
    #             projPath = str((self.projectsPath/dir.name).absolute())
    #             self.projs.append(projPath)
    #             projImportArgs = ['project', 'import', '-i', datasetPath, '-o', projPath, '-f', 'coco', '--overwrite']
    #             main(projImportArgs)
    # def merge(self):
    #     merge_args = ['merge', '-o', 'mergedDataset', '--overwrite', *self.projs]
    #     main(merge_args)
    #     return self
# convert_args=['convert','-i','save_test/task_20201107_seg_토사퇴적-2020_11_19_13_47_08-coco 1.0/','-f','coco','-o','save_test/convert_coco','-if','datumaro']
# main(convert_args)

# add_args=['source','add','path','dumped_COCO/jsonfiles/이음부-단차.json','-f','coco_instances','-p','save_test/task_20201107_bbox_이음부-손상-2020_11_25_10_39_41-coco 1.0/']
# main(add_args)


