import numpy as np
import cv2 as cv
import sys

def Object_tracking(frame1,frame2):
    try:
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    except cv.error:
        sys.exit()

    grayfin = cv.absdiff(gray2, gray1)


    ret, tresh = cv.threshold(grayfin, 45, 255, cv.THRESH_BINARY)

    height, width = frame1.shape[:2]

    contours, hier = cv.findContours(tresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    validated = []

    for cntr in contours:
        x, y, w, h = cv.boundingRect(cntr)
        if x <= width and y >= (height//3) and cv.contourArea(cntr) >= 25:
            validated.append(cntr)
            cv.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),1)

    #cv.drawContours(frame1, validated, -1, (0, 0, 255), 1)
    cv.line(frame1, (0, height//3), (width, height//3), (0, 255, 0), 1)
    cv.imshow("Car_Tracking",frame1)
    cv.imshow("Car_Tracking_Vision",grayfin)

    if cv.waitKey(100) == ord("q"):
        sys.exit()

video = cv.VideoCapture("Cars480.mp4")

while True:
    ret,frame = video.read()
    ret2,frame2 = video.read()

    if not ret and not ret2:
        break

    Object_tracking(frame,frame2)

video.release()
cv.destroyAllWindows()