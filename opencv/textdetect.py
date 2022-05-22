import cv2 as cv
from pytesseract import pytesseract
import sys

pic = cv.imread("plate4.jpg")
bg_remove = cv.createBackgroundSubtractorMOG2(history=100,varThreshold=50) #Create bgsubstractor Instance
cascade = cv.CascadeClassifier()

cascade.load("haarcascades/haarcascade_russian_plate_number.xml")

def refinement(frame):
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # Grayscale
    frame_blurred = cv.GaussianBlur(frame_gray,(9,9),1) # Sharpenning the img
    substracted = bg_remove.apply(frame_blurred) # substract the background from it

    ret,thresh = cv.threshold(substracted,50,255,cv.THRESH_BINARY) # Binarize the img

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,1)) # Create adaptive kernel

    dilated = cv.dilate(thresh,kernel,iterations=1) # Dilate the img

    return dilated

def edge_detection(frame):
    conts,hier = cv.findContours(frame,cv.RETR_TREE,cv.CHAIN_APPROX_NONE) #Detect Contours

    for cnt in conts:
        x,y,w,h = cv.boundingRect(cnt) # Make Contours into a rectangle
        if cv.contourArea(cnt) >= 800: # Sort out false positives
            cv.rectangle(pic,(x,y),(x+w,y+h),(0,0,255),1) #Make a rectangle if it's a car
            plate_detection(x,y,w,h)

def plate_detection(x,y,w,h):
    img = pic[y:y+h,x:x+w]

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_blurred = cv.GaussianBlur(img_gray, (7, 7), 0)

    car_rect = cascade.detectMultiScale(img_blurred, 1.1, 3)

    x, y, w, h = 0, 0, 0, 0

    for (x, y, w, h) in car_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    roi = img[y:y + h, x:x + w]

    roi_blurred = cv.bilateralFilter(roi, 17, 17, 9)

    roi_gray = cv.cvtColor(roi_blurred, cv.COLOR_BGR2GRAY)

    roi_gray = cv.equalizeHist(roi_gray)

    ret, thresh = cv.threshold(roi_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    dilated = cv.dilate(thresh, (7, 7), iterations=3)

    dilated = cv.GaussianBlur(dilated, (5, 5), 1)

    boxes = pytesseract.image_to_boxes(dilated)

    for box in boxes.splitlines():
        box = box.split(" ")
        cv.rectangle(roi, (int(box[1]), int(box[2])), (int(box[1]) + int(box[3]), int(box[2]) + int(box[4])),
                     (0, 255, 0), 2)

    text = pytesseract.image_to_string(dilated, config="-l eng --oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRTUVWXYZ1234567890")

    print(text)

    display(dilated)

def display(pic):
    cv.imshow("picture",pic)
    cv.waitKey(0)
    sys.exit()

pic_refined = refinement(pic)

edge_detection(pic_refined)

