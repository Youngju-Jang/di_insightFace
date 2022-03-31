from retinaface import RetinaFace
from glob import glob
from PIL import Image

# xmin ymin xmax ymax >> x y w h로
def convert(size, box): #size=[w,h], box=[xyxy]
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = round(x*dw,8)
    w = round(w*dw,8)
    y = round(y*dh,8)
    h = round(h*dh,8)
    return [x,y,w,h]

def get_size(img_path):
    image = Image.open(img_path)
    w = int(image.size[0])
    h = int(image.size[1])
    return [w, h]

folder_path = "\dataset\dataTest\images\\"

# 이미지리스트 뽑기
ori_paths = glob(folder_path + '*.jpg')

# x1 y1 x2 y2 facial_area 좌표뽑기
for ori_path in ori_paths : #이미지 하나당
    ori_file = ori_path.split("\\")[-1] # 파일명.jpg
    ori_dict = RetinaFace.detect_faces(ori_path)
    size = get_size(ori_path)

    with open('/dataset/dataTest/labels/'+ori_file[:-4]+'.txt','w') as f:
        # 한장안에 한사람씩
        for face_n in ori_dict: #얼굴하나당
            xyxy = ori_dict[face_n]['facial_area'] # [x1 y1 x2 y2]
            # yolo form으로 변경
            xywh = convert(size, xyxy)
            ## 한장에 여러명일경우로 수정
            # txt화
            f.write(" ".join(str(_) for _ in xywh))





print("end")



'''
len(obj.keys()) >> 사진에 얼굴수대로 나옴 >> 한명이라 face_1

print(obj['face_1'])
{'score': 0.9993307590484619, 'facial_area': [168, 145, 346, 390], x1 y1 x2 y2 일듯? 
'landmarks': {'right_eye': [216.07236, 250.7787], 'left_eye': [298.8688, 240.17404], 
'nose': [265.16446, 292.47305], 'mouth_right': [234.40356, 336.24783], 
'mouth_left': [302.9504, 326.35788]}}
'''