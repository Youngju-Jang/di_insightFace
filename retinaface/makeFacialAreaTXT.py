from retinaface import RetinaFace
from glob import glob

folder_path = "/dataset/dataTest/ori/"
ori_path = glob(folder_path + '*.jpg')
print(ori_path)

# face bounding box
#obj = RetinaFace.detect_faces(img_path)

#print(obj)
'''
len(obj.keys()) >> 사진에 얼굴수대로 나옴. 한명이라 face_1

print(obj['face_1'])
{'score': 0.9993307590484619, 'facial_area': [168, 145, 346, 390], x1 y1 x2 y2 일듯? 
'landmarks': {'right_eye': [216.07236, 250.7787], 'left_eye': [298.8688, 240.17404], 
'nose': [265.16446, 292.47305], 'mouth_right': [234.40356, 336.24783], 
'mouth_left': [302.9504, 326.35788]}}
'''