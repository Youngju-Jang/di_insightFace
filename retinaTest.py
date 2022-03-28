from retinaface import RetinaFace

img_path = "/dataset/dataTest/test1.jpg"

obj = RetinaFace.detect_faces(img_path)

obj.keys()