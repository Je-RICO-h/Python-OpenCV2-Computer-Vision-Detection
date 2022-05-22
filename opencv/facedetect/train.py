import cv2 as cv
import numpy as np
from sys import exit
import os

def trainmodel():

    ppl = ["Ben Afflek","Elton John","Jerry Seinfield","Madonna","Mindy Kaling"]

    face_cascade = cv.CascadeClassifier()
    eye_cascade = cv.CascadeClassifier()

    try:
        face_cascade.load("E:/Python_programs/opencv/haarcascades/haarcascade_frontalface_alt.xml")
        eye_cascade.load("E:/Python_programs/opencv/haarcascades/haarcascade_eye.xml")
    except:
        exit("Cascades could not be opened!")

    features = []
    labels = []

    def facedetect_train():
        for person in ppl:

            imgfolder = f"E:/Python_programs/opencv/Faces/train/{person}"
            label = ppl.index(person)

            for imgnum in os.listdir(imgfolder):

                imgpath = f"{imgfolder}/{imgnum}"

                img = cv.imread(imgpath)

                img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                img_grey = cv.equalizeHist(img_grey)

                faces = face_cascade.detectMultiScale(img_grey, scaleFactor=1.1,minNeighbors=4)

                for (x,y,w,h) in faces:
                    face_roi = img_grey[y:y+h,x:x+w]
                    features.append(face_roi)
                    labels.append(label)

    facedetect_train()

    features = np.array(features,dtype="object")
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    face_recognizer.train(features,labels)

    face_recognizer.save('face_trained.yml')
    np.save('E:/Python_programs/opencv/training/features.npy',features)
    np.save('E:/Python_programs/opencv/training/labels.npy',labels)

    print("Training Complete!")

