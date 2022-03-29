import os 
import sys
import cv2

#sys.path.append('c:/users/joj10/anaconda3/envs/apple/lib/site-packages')
#from RetinaFace import RetinaFace

from retinaface import RetinaFace
print(sys.path)
1+111

'''
img_path = "/dataset/dataTest/test1.jpg"

obj = RetinaFace.detect_faces(img_path)

obj.keys()


# init with normal accuracy option
detector = RetinaFace(quality="normal")

# same with cv2.imread,cv2.cvtColor 
rgb_image = detector.read("data/bus.jpg")

faces = detector.predict(rgb_image)
# faces is list of face dictionary
# each face dictionary contains x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
# faces=[{"x1":20,"y1":32, ... }, ...]
print(faces)

result_img = detector.draw(rgb_image,faces)

# save ([...,::-1] : rgb -> bgr )
cv2.imwrite("data/result_img.jpg",result_img[...,::-1])

# show using cv2
# cv2.imshow("result",result_img[...,::-1)
# cv2.waitKey()
'''