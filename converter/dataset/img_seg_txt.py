import os
'''
img_file1 segmentation_file1
img_file2 segmentation_file2
..
..

이런 모양의 txt파일을 만듦
'''
def mkdataset(txtpath, savepath,dataset='voc'):
    with open(txtpath, 'r') as f:
        datasettxt = f.readlines()
    datasettxt = [i.rstrip('\n') for i in datasettxt]

    if dataset=='voc':
        # root = os.path.join(os.path.dirname(path),'../..')
        # root = os.path.normpath(root)
        imgpath = [os.path.join('JPEGImages',f'{i}.jpg') for i in datasettxt]
        labelpath = [os.path.join('SegmentationClass',f'{i}.png') for i in datasettxt]
        mergepath = list(zip(imgpath, labelpath))
        mergepath = [' '.join(i) for i in mergepath]

    with open(savepath, 'w') as f:
        f.write('\n'.join(mergepath))
    filename = os.path.basename(txtpath)
    print(f'{filename}\tsave done!')

folderpath = '/home/tm/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation'
savefolder = '/home/tm/Code/semseg/list/voc2012'
txtlist = os.listdir(folderpath)

for i in txtlist:
    txtpath = os.path.join(folderpath,i)
    savepath = os.path.join(savefolder,i)
    mkdataset(txtpath,savepath)


# mkdataset('/home/tm/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt',
#             '/home/tm/Code/semseg/list/voc2012/train.txt')