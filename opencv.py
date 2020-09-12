import cv2
import os

# scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
# This scale factor is used to create scale pyramid as shown in the picture. ...
# This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality.

# minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
# In other words, this parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality.
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 1, color['blue'], "Face")
    # if len(coords) == 4:
    #     roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
    coords = draw_boundary(img, eyeCascade, 1.2, 2, color['red'], "Eye")
    coords = draw_boundary(img, noseCascade, 1.3, 3, color['green'], "Nose")
    coords = draw_boundary(img, mouthCascade, 1.3, 6, color['white'], "Mouth")
    return img
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# os.chdir('/home/archit/Downloads/Mask_ No_mask/Face Mask Dataset/Train/WithoutMask')
# img = cv2.imread('126.png')
#
# img = cv2.resize(img,(224,224))
# img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
# cv2.imshow('IMAGE', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


video_capture = cv2.VideoCapture(3)

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()