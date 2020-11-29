from pathlib import Path
# file__ = Path(__file__).parent
import sys
# sys.path.insert(0,str(file__/'datumaro'))

from datumaro.cli.__main__ import main
import os
import stat
from easydict import EasyDict
from typing import Callable, OrderedDict,Dict
from shutil import rmtree

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
        if Path('tmp').is_dir():
            rmtree(str(Path('tmp').absolute()), onerror=remove_readonly)
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

class CLI:
    def __init__(self) -> None:
        self.datasetsPath = Path('datasets').absolute()
        self.datasetsPath.mkdir(exist_ok=True)
        self.projectsPath = (Path('tmp/projects')).absolute()

    def importDataset(self, args):
        self.projectsPath.mkdir(exist_ok=True, parents=True)
        datasetPathList = self.getSubDirList(self.datasetsPath)
        for datasetPath in datasetPathList:
            projPath = str(self.projectsPath/datasetPath.name)
            projImportArgs = ['project', 'import', '-i', str(datasetPath), '-o', projPath, '-f', args.format.lower(), '--overwrite']
            main(projImportArgs)
        return self

    def mergeDataset(self):
        projsPathList = self.getSubDirList(self.projectsPath)
        projsPathList = [str(dir) for dir in projsPathList]
        mergePath = (self.projectsPath/'merged')
        mergePath.mkdir(exist_ok=True)
        merge_args = ['merge', '-o', str(mergePath), '--overwrite', *projsPathList]
        main(merge_args)
        return self

    def exportDataset(self, args, merge=False):
        if merge:
            projectsPathList = [self.projectsPath/'merged']
        else:
            projectsPathList = self.getSubDirList(self.projectsPath)
        for proj in projectsPathList:
            exportPath = (Path('exportDataset')/args.format.lower()/proj.name).absolute()
            exportPath.mkdir(exist_ok=True, parents=True)
            export_args = ['project','export','-f',args.format.lower(),'-o',str(exportPath),'-p',str(proj)]
            main(export_args)
        rmtree(str(Path('tmp').absolute()), ignore_errors=True, onerror=remove_readonly)
        return self

    def setArg(self, argsAndType:Dict[str,str]=dict())->EasyDict:
        args = EasyDict()
        for name,(type, help) in argsAndType.items():
            inputPrint = f'\n\n{help}\n{name} 입력: '
            val = input(inputPrint)
            arg = eval(f'{type}(\'{val.strip()}\')')
            args[name] = arg
        return args

    def getSubDirList(self,Path:Path):
        return [dir for dir in Path.iterdir() if dir.is_dir()]

    def mergeFunction(self):
        importArgs = self.setArg({'format':('str', '다운로드 한 데이터셋의 형식.\n지원형식: coco, cvat, datumaro, image_dir, imagenet, imagenet_txt, label_me, mot_seq, mots,tf_detection_api, voc, yolo')})
        exportArgs = self.setArg({'format':('str', '내보낼 데이터셋의 형식.\n지원형식:coco, cvat, datumaro, datumaro_project, label_me, mot_seq_gt, mots_png, tf_detection_api, voc,voc_segmentation, yolo')})
        self.importDataset(importArgs)
        self.mergeDataset()
        self.exportDataset(exportArgs, merge=True)
        return self

    def convertFunction(self):
        importArgs = self.setArg({'format':('str', '다운로드 한 데이터셋의 형식.\n지원형식: coco, cvat, datumaro, image_dir, imagenet, imagenet_txt, label_me, mot_seq, mots,tf_detection_api, voc, yolo')})
        exportArgs = self.setArg({'format':('str', '내보낼 데이터셋의 형식.\n지원형식:coco, cvat, datumaro, datumaro_project, label_me, mot_seq_gt, mots_png, tf_detection_api, voc,voc_segmentation, yolo')})
        self.importDataset(importArgs)
        self.exportDataset(exportArgs)
        return self

if __name__ == "__main__":
    command = CLI()
    mainMenu = Menu('Main').setCommand('dataset 합치기',command.mergeFunction,'여러 데이터셋을 하나로 합칩니다.')\
                        .setCommand('dataset 변환', command.convertFunction, '다른 형식의 데이터셋으로 바꿉니다. 예) coco format -> voc format')

    mainMenu()