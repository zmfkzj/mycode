import util
import os
import random

# V&V 데이터에서 정확도 조정을 위해 False Negative가 존재하는 이미지를 삭제
# classes = ['Peeling', 'desqu']
# classes = ["Leakage", 'desqu']
# classes = ["Crack", "Leakage" ,"Peeling", "Desqu", "Fail"]
classes = ["Crack"]
rmimglist = []
for cls in classes:
    for i in range(1,20):
        try:
            rmimglist.extend(util.txt2list(f'/home/tm/output_img/최종발표대비/V&V/no_2_{cls}_{i}.txt'))
        except:
            pass
random.shuffle(rmimglist)
# rmimglist = rmimglist[:int(len(rmimglist)*0.7)]
rmtxtlist = util.chgext(rmimglist,'.txt')
rmxmllist = util.chgext(rmimglist,'.xml')
errorcount = 0
successcount = 0
for paths in zip(rmimglist, rmtxtlist, rmxmllist):
    if successcount == 74:
        break
    try:
        for path in paths:
            os.remove(path)
        successcount +=1
    except:
        errorcount +=1
        pass
print('remove complete : ', successcount)
