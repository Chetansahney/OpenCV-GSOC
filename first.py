import cv2
import numpy as np
import face_recognition as fr

frameWidth=360
frameHeight=360
cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
def empty(a):
    pass
cv2.namedWindow("parameters")
cv2.createTrackbar("threshold1","parameters",150,255,empty)
cv2.createTrackbar("threshold2","parameters",255,255,empty)
def getcontours(img,imgcontours):
        contours,heirarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgcontours,contours,-1,(255,0,255),7)
        
        for cnt in contours:
             area =cv2.contourArea(cnt)
             if area >1000:
               cv2.drawContours(imgcontours,cnt,-1,(255,0,255),7)
               peri=cv2.arcLength(cnt,True)
               approx=cv2.approxPolyDP(cnt,0.02*peri,True)
               print(len(approx))




while True:
    _, img= cap.read()
    imgContour=img.copy()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    threshold1=cv2.getTrackbarPos("threshold1","parameters")
    threshold2=cv2.getTrackbarPos("threshold2","parameters") 
    imgcanny=cv2.Canny(imgGray,threshold1,threshold2)
    
    getcontours(imgcanny,imgContour)
    

    cv2.imshow("original", imgContour)
    key=cv2.waitKey(1)

    if key==27 :
        break

cap.release()
