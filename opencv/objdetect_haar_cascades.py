import cv2 as cv
import numpy as np # Future project

#Load Things
video = cv.VideoCapture("Cars3.mp4")
cascade = cv.CascadeClassifier()

cascade.load("haarcascades/cars.xml")

while True:
    #Read Frame
    ret,frame = video.read()

    if not ret:
        break

    #Info
    # height,width = frame.shape[:2]
    #                                       Cars 2 Material
    # roi = frame[:,280:width-280]



    #Equalize and Grayscale
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.equalizeHist(frame_gray)


    #HaarCascading
    car_rect = cascade.detectMultiScale(frame_gray,scaleFactor=1.1,minNeighbors=3)

    for (x,y,w,h) in car_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=1)

    cv.imshow("Car_detect_haar",frame)
    if cv.waitKey(30) == ord("q"):
        break

video.release()
cv.destroyAllWindows()