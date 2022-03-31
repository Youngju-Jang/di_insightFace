from PIL import Image

def convert(size, box):
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

xy = "/dataset/dataTest/images/test1.jpg"
print(xy.split("/")[-1][:-4])
print(xy[:-4] + ".txt")


li = [168, 145, 346, 390]
print(" ".join(str(_) for _ in li))

im=Image.open(xy)
w= int(im.size[0])
h= int(im.size[1])

print('w, h : ', w, h)
yolo_form = convert( [w,h], li)
print("yolo_form", yolo_form)

