# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:25:51 2020

@author: gaurav sahani
"""


import cv2
import time

# Create our body classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/gaurav sahani/Desktop/Cars.mp4')

# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.05)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1)== 13: #is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()