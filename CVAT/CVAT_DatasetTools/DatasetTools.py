from commands import Menu, CLI

command = CLI()
mainMenu = Menu('Main').setCommand('dataset 합치기',command.mergeFunction,'여러 데이터셋을 하나로 합칩니다.')\
                       .setCommand('dataset 변환', command.convertFunction, '다른 형식의 데이터셋으로 바꿉니다. 예) coco format -> voc format')

mainMenu()