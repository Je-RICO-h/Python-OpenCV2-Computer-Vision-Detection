import cv2 as cv
from pytesseract import pytesseract
import numpy as np
import sys

cascade = cv.CascadeClassifier()

cascade.load("haarcascades/haarcascade_russian_plate_number.xml")

img = cv.imread("plate3.jpg")

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

img_blurred = cv.GaussianBlur(img_gray,(7,7),0)

car_rect = cascade.detectMultiScale(img_blurred,1.1,3)

x,y,w,h = 0,0,0,0

for (x,y,w,h) in car_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

roi = img[y:y+h,x:x+w]

roi_resized = cv.resize(roi,None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)

roi_blurred = cv.bilateralFilter(roi_resized,17,17,9)

roi_gray = cv.cvtColor(roi,cv.COLOR_BGR2GRAY)

roi_gray = cv.equalizeHist(roi_gray)

ret,thresh = cv.threshold(roi_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

dilated = cv.dilate(thresh,(5,5),iterations=1)

dilated = cv.GaussianBlur(dilated,(1,1),1)

boxes = pytesseract.image_to_boxes(dilated)

for box in boxes.splitlines():
    box = box.split(" ")
    cv.rectangle(roi,(int(box[1]),int(box[2])),(int(box[1])+int(box[3]),int(box[2])+int(box[4])),(0,255,0),2)

text = pytesseract.image_to_string(dilated, config="-l eng --psm 9 _char_whitelist=ABCDEFGHIJKLMNOPQRTUVWXYZ1234567890")

print(text)

cv.imshow("car",dilated)
cv.imshow("car2",img)

cv.waitKey(0)

cv.destroyAllWindows()