import cv2 as cv
import sys

video = cv.VideoCapture("Cars480.mp4") # Load Video
bg_remove = cv.createBackgroundSubtractorMOG2(history=100,varThreshold=50) #Create bgsubstractor Instance

#import haarcascade

cascade = cv.CascadeClassifier()

cascade.load("haarcascades/haarcascade_russian_plate_number.xml")

def display(frame,wait=False):
    cv.imshow("frame", frame)

    if wait:
        time = 0 #Picture
    else:
        time = 30 #Video

    if cv.waitKey(time) == ord("q"):
        sys.exit()

def refinement(frame):
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # Grayscale
    frame_blurred = cv.GaussianBlur(frame_gray,(9,9),1) # Sharpenning the img
    substracted = bg_remove.apply(frame_blurred) # substract the background from it

    ret,thresh = cv.threshold(substracted,50,255,cv.THRESH_BINARY) # Binarize the img
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,1)) # Create adaptive kernel

    dilated = cv.dilate(thresh,kernel,iterations=1) # Dilate the img

    return dilated

def edge_detection(oframe,frame):
    conts,hier = cv.findContours(frame,cv.RETR_TREE,cv.CHAIN_APPROX_NONE) #Detect Contours

    for cnt in conts:
        x,y,w,h = cv.boundingRect(cnt) # Make Contours into a rectangle
        if cv.contourArea(cnt) >= 800: # Sort out false positives
            cv.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),1) #Make a rectangle if it's a car
            plate_detection(oframe,x,y,w,h)

def plate_detection(frame,x,y,w,h):
    plate = frame[y:y + h, x:x + w] # Cut out car area

    plate_gray = cv.cvtColor(plate, cv.COLOR_BGR2GRAY)  #IMG_manip

    plate_blurred = cv.GaussianBlur(plate_gray, (7, 7), 0)

    plate_rect = cascade.detectMultiScale(plate_blurred, 1.1, 3) #Detect Plate

    x, y, w, h = 0, 0, 0, 0

    for (x, y, w, h) in plate_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1) # Draw out plate

while True:
    #Read Video frame by frame
    cap,frame = video.read()

    if not cap:
        break

    #roi = frame[100:, 100:900]  # Range of interest Cars3
    roi = frame[:,:] # Roi Cars480

    #Call functions and display

    frame_final = refinement(roi)

    edge_detection(frame,frame_final)

    display(frame,False)

#Release Memory and terminate program

video.release()
cv.destroyAllWindows()