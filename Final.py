import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'

print('Loading known faces')
known_faces = []
known_names = []

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

video = cv2.VideoCapture(2)
color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouthCascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    _, image = video.read()
    print("before")
    # coords = draw_boundary(image, noseCascade, 1.3, 3, color['green'], "Nose")
    # coords = draw_boundary(image, mouthCascade, 1.3, 6, color['white'], "Mouth")
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print('after')
    locations = face_recognition.face_locations(image, model = MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    coords = draw_boundary(image, faceCascade, 1.1, 1, color['blue'], "Face")
    if len(coords) == 4:
        roi_img = image[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        coords = draw_boundary(roi_img, eyesCascade, 1.2, 2, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.3, 3, color['green'], "Nose")
        coords = draw_boundary(roi_img, mouthCascade, 1.3, 6, color['white'], "Mouth")
    print("safv")
    for face_encoding, face_location in zip(encodings, locations):
        print("in loop")
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        # coords = draw_boundary(image, noseCascade, 1.3, 3, color['green'], "Nose")
        # coords = draw_boundary(image, mouthCascade, 1.3, 6, color['white'], "Mouth")
        match = None
        if True in results:
            match = known_names[results.index(True)]
            top_left = (face_location[3], face_location[0])
            # roi_image = image[face_location[1]:face_location[3], face_location[0]:face_location[2]]

            bottom_right = (face_location[1], face_location[2])
            # color = [0, 255, 0]
            # cv2.rectangle(image, top_left, bottom_right, color['blue'], FRAME_THICKNESS)
    #
    #         # top_left = (face_location[3], face_location[0])
    #         # bottom_right = (face_location[1], face_location[2]+22)
    #         # cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color['red'], FONT_THICKNESS)
    #         # roi_image = image[face_location[1]:face_location[3], face_location[0]:face_location[2]]
    #         # coords = draw_boundary(image, noseCascade, 1.3, 3, color['green'], "Nose")
    #         # coords = draw_boundary(image, mouthCascade, 1.3, 6, color['white'], "Mouth")

    cv2.imshow('camera', image)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    else:
        continue

video_capture.release()
cv2.destroyAllWindows()