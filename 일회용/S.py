import numpy as np

'''
폭파구 면적 구하기
'''

predict = (0,0,5,5) #검출 결과

x1, y1, x2, y2 = predict # 검출결과를 x1, y1, x2, y2로 매핑

S_box = (x2-x1)*(y2-y1) #bounding box의 넓이
S_crater = np.pi/4*S_box #폭파구의 넓이

print(f'{S_box=}')
print(f'{S_crater=}')
print(f'{0.7854*S_box=}')


