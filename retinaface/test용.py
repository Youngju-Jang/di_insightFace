from retinaface import RetinaFace
img_path = "\dataset\dataTest\images\\test1.jpg"
ori_dict = RetinaFace.detect_faces(img_path)

print("ori_dict>> ",ori_dict['face_1'])
xyxy = ori_dict['face_1']['facial_area'] # [x1 y1 x2 y2]
print(xyxy)
