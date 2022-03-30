import matplotlib.pyplot as plt
from retinaface import RetinaFace

#print(os.getcwd())

img_path = "/dataset/dataTest/test1.jpg"

faces = RetinaFace.extract_faces(img_path= img_path, align= True)
for face in faces:
    plt.imshow(face)
    plt.show()