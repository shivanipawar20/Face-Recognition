import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('C:/python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor = 1.25, minNeighbors = 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (100,0,250), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = img[y:y + h, x:x + w]
    cv2.imshow('Face Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: #escape key
        break

cap.release()
cv2.destroyAllWindows()
