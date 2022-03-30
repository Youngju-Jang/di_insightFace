import matplotlib.pyplot as plt
from retinaface import RetinaFace
import sys

#print(os.getcwd())

img_path = "/dataset/dataTest/ori/test1.jpg"

faces = RetinaFace.extract_faces(img_path= img_path, align= True)

print(faces)

sys.exit()

for face in faces:
    plt.imshow(face)
    plt.show()