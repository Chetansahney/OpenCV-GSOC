# -*- coding: utf-8 -*-
import face_recognition as fr
import cv2
import numpy as np
imgelon=fr.load_image_file("resources/elon.jpg")
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest=fr.load_image_file("resources/elon2.jpg")
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
facloc=fr.face_locations(imgelon)[0]
encodeelon=fr.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)

facloctest=fr.face_locations(imgtest)[0]
encodeelontest=fr.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(facloctest[3],facloctest[0]),(facloctest[1],facloctest[2]),(255,0,255),2)
#linear svm is used for face recognition
result=fr.compare_faces([encodeelon],encodeelontest)
facedistance=fr.face_distance([encodeelon],encodeelontest) #lower distance=better match
cv2.putText(imgtest,f'{result} {round(facedistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)


cv2.imshow("elon",imgelon)
cv2.imshow("elon2",imgtest)
cv2.waitKey(0)
cv2.destroyAllWindows()