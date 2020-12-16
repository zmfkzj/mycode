from sklearn.model_selection import train_test_split
from pathlib import Path
from copy import deepcopy

def train_valid_test_split(defaultTxtFilePath):
    #defualt.txt를 train, val, trainval, test 나눔
    defaultPath = Path(defaultTxtFilePath)
    with open(defaultTxtFilePath, 'r', encoding='utf-8') as f:
        images = f.readlines()
    images = [line.rstrip('\n') for line in images]

    train_img, testval_img = train_test_split(images, test_size=0.4, random_state=31)
    val_img, test_img = train_test_split(testval_img, test_size=0.5, random_state=31)
    trainval_img = deepcopy(train_img)
    trainval_img.extend(val_img)

    #save txt
    list2txt(train_img, str( defaultPath.parent/'train.txt'))
    list2txt(val_img, str(defaultPath.parent/'val.txt'))
    list2txt(trainval_img, str(defaultPath.parent/'trainval.txt'))
    list2txt(test_img, str(defaultPath.parent/'test.txt'))


def list2txt(List, savepath, mode='w'):
    with open(savepath,mode, encoding='utf-8') as f:
        f.write("\n".join(List))

if __name__ == "__main__":
    train_valid_test_split('E:\\default.txt')