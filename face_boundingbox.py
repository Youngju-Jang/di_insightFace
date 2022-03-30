import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace

img_path = "/dataset/dataTest/ori/test1.jpg"
img = cv2.imread(img_path)
# face bounding box
obj = RetinaFace.detect_faces(img_path)

'''
len(obj.keys()) >> 사진에 얼굴수대로 나옴. 한명이라 face_1

print(obj['face_1'])
{'score': 0.9993307590484619, 'facial_area': [168, 145, 346, 390], x1 y1 x2 y2 일듯? 
'landmarks': {'right_eye': [216.07236, 250.7787], 'left_eye': [298.8688, 240.17404], 
'nose': [265.16446, 292.47305], 'mouth_right': [234.40356, 336.24783], 
'mouth_left': [302.9504, 326.35788]}}
'''
#사진 한장안에 사람얼굴 수만큼 bb그리기
for key in obj.keys():
    identity = obj[key]

    facial_area = identity['facial_area']

    #cv2.rectangle(img, pt1(x,y:시작점), pt2(x,y:종료점), color, thickness)
    cv2.rectangle(img, (facial_area[2], facial_area[3]),(facial_area[0], facial_area[1]),(255, 255, 255), 1)

plt.figure(figsize = (20, 20))
plt.imshow(img[:,:,: :-1])
plt.show()
