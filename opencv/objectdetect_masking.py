import cv2 as cv

#Import Video
video = cv.VideoCapture("Cars2.mp4")
bgremove = cv.createBackgroundSubtractorMOG2(history=240,varThreshold=70)

while True:
    #Read Frame
    ret,frame = video.read()

    if not ret:
        break

    #Cleaning
    sharpened_frame = cv.bilateralFilter(frame,5,50,50)

    #Masking
    mask = bgremove.apply(sharpened_frame) #Masking
    cv.line(frame,(350,200),(600,200),(0,0,255),2) # ROI Line

    #Thresholding
    thresh = cv.adaptiveThreshold(mask,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv.THRESH_BINARY,11,2)

    #Find Contours and display found car

    contours, hier = cv.findContours(thresh.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    validated_contours = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (x <= frame.shape[0] and x >= 350) and y >= 200 and cv.contourArea(cnt) >= 50:
            validated_contours.append(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    #Display
    cv.imshow("CarDetect",frame)
    cv.imshow("CarDetect_Masked",mask)
    if cv.waitKey(30) == ord("q"):
        break

video.release()
cv.destroyAllWindows()