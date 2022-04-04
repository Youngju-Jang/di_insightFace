import matplotlib.pyplot as plt
from retinaface import RetinaFace
import cv2
import os

#print(os.getcwd())

img_path = "/dataset/dataTest/images/test1.jpg"
img = img_path.split("/")[-1][:-4]
print(img)

faces = RetinaFace.extract_faces(img_path= img_path, align= True)

print(faces)


for face in faces:
    plt.imshow(face)
    plt.show()

for face in faces:
    cv2.imshow('bb_img', face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite()