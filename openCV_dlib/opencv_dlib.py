import numpy as np
import cv2
import dlib
import os
import imutils
from glob import glob

# https://ichi.pro/ko/opencv-dlibleul-sayonghan-anmyeon-maseukeu-obeolei-85300355583053
## set directories
os.chdir('C:\\Users\joj10\PycharmProjects\insightFace\openCV_dlib')

folder_path = 'C:\dataset\\noMask\wiki_crop\\00\\'
imgs_path = glob(folder_path + '*.jpg')

# Initialize dlib's shape predictor
dat_path = 'C:\\Users\joj10\PycharmProjects\insightFace\dat_dir\\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(dat_path)

# initialize color [color_type] = (blue, green, red) 순서주의
color_blue = (239,207,137)
color_cyan = (255,200,0)
color_black = (0, 0, 0)


# 사진한장 안에 사람 한명 face 의 landmark의 mask type값 설정
def make_mask_mark(type): # type = 1,2,3
    landmarks = predictor(gray, face)
    points = []

    # fmask_a = wide, high coverage mask,
    # fmask_c = wide, medium coverage mask, >>이거사용
    # fmask_e = wide, low coverage mask

    for i in range(1, 16): #하관쪽 쭉둘르기
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)

    # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
    mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
              ((landmarks.part(27).x), (landmarks.part(27).y)),
              ((landmarks.part(39).x), (landmarks.part(1).y))]

    # Coordinates for the additional point for wide, medium coverage mask - in sequence
    mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))] #코중앙

    # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
    mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
              ((landmarks.part(34).x), (landmarks.part(34).y)),
              ((landmarks.part(33).x), (landmarks.part(33).y)),
              ((landmarks.part(32).x), (landmarks.part(32).y)),
              ((landmarks.part(31).x), (landmarks.part(31).y))]

    fmask_a = points + mask_a
    fmask_c = points + mask_c
    fmask_e = points + mask_e

    mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}

    return mask_type[type]

# 이미지에 마스크 그리고 색채우기
def fill_img_mask(img, color, fmask):

    # Using Python OpenCV – cv2.polylines() method to draw mask outline for [mask_type]:

    fmask = np.array(fmask, dtype=np.int32)

    # change parameter [mask_type] and color_type for various combination
    img2 = cv2.polylines(img, [fmask], True, color, thickness=2, lineType=cv2.LINE_8)

    # Using Python OpenCV – cv2.fillPoly() method to fill mask
    # change parameter [mask_type] and color_type for various combination
    img3 = cv2.fillPoly(img2, [fmask], color, lineType=cv2.LINE_AA)

    return img3


# 이미지 전 처 리
for img_path in imgs_path[:5]:
    img_file = img_path.split('\\')[-1] #파일명.jpg

    img = cv2.imread(img_path)
    img = imutils.resize(img, width = 500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1) # rectangles[[(167, 142) (390, 365)]]
    #print("Number of faces detected: ", len(faces))

    for face in faces:
        ''' 
        # (x1,x2,y1,y2) 만들기
        # Using a for loop in order to extract the specific coordinates (x1,x2,y1,y2)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # Drawing a rectangle around the face detected
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        '''

        ''' 
        # Get the shape using the predictor
        landmarks = predictor(gray, face)
        #68개 다찍기
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            img_landmark = cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        '''
        # 마스크부분 점위치
        fmask = make_mask_mark(2)

        # 이미지에 마스크그리기
        img = fill_img_mask(img, color_blue, fmask)

    '''
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # Save the output file for testing
    outputNameofImage = "C:\dataset\Mask\\" + img_file
    print("Saving output image to", outputNameofImage)
    cv2.imwrite(outputNameofImage, img)