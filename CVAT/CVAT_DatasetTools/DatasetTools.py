from pathlib import Path
file__ = Path(__file__)
import sys
sys.path.insert(0,'E:\\merge\\datumaro_')

# from datumaro.components.project import Project
from datumaro.cli.__main__ import main
import os
from easydict import EasyDict
from typing import Callable
import inspect

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
            line = f'{val.no}\t\t{val.opt}\t{val.help}'
            opts.append(line)
        return '\n'.join(opts)

    def selectAndRun(self):
        self.prevMunu = Menu.prevMunu
        Menu.prevMunu = self
        consolClear()
        print(self)
        selNum = int(input('선택 No: ').strip())
        self.command[selNum].function(*self.command.argValues)

    def setCommand(self, opt:str, function:Callable, argNames:dict[str,str]=dict(), help='') :
        self.command[len(self.command)] = {"opt": opt,
                                        "function": function, 
                                        "argNames": argNames, 
                                        "argValues": [], 
                                        "help": help}
        return self

    def setArg(self,no:str):
        args = []
        for name,type in self.command[no].argNames.items():
            val = input(f'{name}:{type} 입력: ')
            arg = eval(f'{type}({val.strip()})')
            args.append(arg)

        self.command[no].argValues = args
        return self

    def goToPrev(self):
        self.prevMunu.selectAndRun()

class CLI:
    def __init__(self, datasetsPath) -> None:
        self.datasetsPath = Path(datasetsPath).absolute()
        self.datasetPathList = []
        self.projectsPath = (Path('projects')/self.datasetPath.name).absolute()
        self.exportPath = (Path('export')/self.datasetPath.name).absolute()

    def importDataset(self, format):
        datasetPath = str(self.datasetPath)
        projPath = str(self.projectsPath)
        projImportArgs = ['project', 'import', '-i', datasetPath, '-o', projPath, '-f', format, '--overwrite']
        main(projImportArgs)
        return self

    def exportDataset(self, format):
        export_args = ['project','export','-f',format,'-o','export_test_voc','-p','merge_test/']
        main(export_args)

mainMenu = Menu('Main').setCommand('dataset 합치기',)


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


