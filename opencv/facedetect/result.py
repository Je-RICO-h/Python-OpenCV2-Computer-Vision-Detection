from train import trainmodel
import cv2 as cv
import numpy as np

trainmodel()

ppl = ["Ben Afflek","Elton John","Jerry Seinfield","Madonna","Mindy Kaling"]

features = np.load('E:/Python_programs/opencv/training/features.npy', allow_pickle=True)
labels = np.load('E:/Python_programs/opencv/training/labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

face_cascade = cv.CascadeClassifier()
eye_cascade = cv.CascadeClassifier()

try:
    face_cascade.load("E:/Python_programs/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    eye_cascade.load("E:/Python_programs/opencv/haarcascades/haarcascade_eye.xml")
except:
    exit("Cascades could not be opened!")

img = cv.imread(f"E:/Python_programs/opencv/Faces/val/jerry_seinfeld/3.jpg")

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_rect = face_cascade.detectMultiScale(img_gray,1.1,4)

for (x, y, w, h) in face_rect:
    face_roi = img_gray[y:y + h, x:x + w]

    label,confidence = face_recognizer.predict(face_roi)
    print(f'Index = {ppl[label]} with matching: {round(confidence)}%')

    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
    cv.putText(img, ppl[label],(x,y+w),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    eyes = eye_cascade.detectMultiScale(face_roi,1.1,4)
    for (x2,y2,w2,h2) in eyes:
        eye_center = (x+x2+w2//2,y+y2+h2//2)
        radius = int(round((w2+h2)*0.25))
        cv.circle(img,eye_center,radius,(0,255,0),2)

cv.imshow('FaceDetection',img)

cv.waitKey(0)