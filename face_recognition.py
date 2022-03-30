import cv2
from deepface import DeepFace

img_path = "/dataset/dataTest/ori/test1.jpg"
img = cv2.imread(img_path)

# face recognition
# 사진두개 너혹, 결과돌리면 obj['verified']값이 true or false 나올거
obj = DeepFace.verify(img1_path=img_path, img2_path='img2.jpg', model_name='ArcFace', detector_backend='retinaface')
