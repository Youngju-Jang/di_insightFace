import cv2
from retinaface import RetinaFace

img_path = "\dataset\dataTest\images\\test1.jpg"
label_path = "\dataset\dataTest\labels\\test1.txt"
ori_dict = RetinaFace.detect_faces(img_path)
xyxy = ori_dict['face_1']['facial_area']

img = cv2.imread(img_path, 1)

cv2.rectangle(img, (xyxy[0],xyxy[1]), (xyxy[2], xyxy[3]), (0,255), 3)

cv2.imshow('bb_img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

# 이미지 저장하기
cv2.imwrite('저장될파일명', '저장할이미지')

