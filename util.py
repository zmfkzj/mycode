import os

def txt2list(txtpath):
    #텍스트 파일의 내용을 리스트로 만들기
    with open(txtpath, 'r') as f:
        filelist = f.readlines()
    filelist = [path.rstrip('\n') for path in filelist]
    filelist = [path for path in filelist if path]
    return filelist

def chgext(pathlist, newext):
    '''
    리스트 각 원소의 확장자 변경
    ex) newext = '.xml'
    '''
    newpathlist = []
    for path in pathlist:
        other, _ = os.path.splitext(path)
        newpathlist.append(f'{other}{newext}')
    return newpathlist

def list2txt(List, savepath):
    with open(savepath,'w') as f:
        f.write("\n".join(List))

def ChkNMkFolder(folderpath):
    #folderpath에 폴더 만들기. 중간에 없는 폴더도 만듬
    if not os.path.isdir(folderpath):
        os.makedirs(folderpath)

def pickFilename(path):
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    return filename

if __name__ == "__main__":
    ChkNMkFolder("/home/tm/Code/mycode/b/c")
