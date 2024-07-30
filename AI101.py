
import cv2 as cv
import numpy as np
import pickle as p

# face recognition project



labels = {}
            
with open ('j.pickle', 'rb') as f:
    labels = p.load(f)
    reversed_label = {v: k for k, v in labels.items()}

# opens the camera
# 1
vid  = cv.VideoCapture(0)


# opens the haar cascade
# note: the XML file was downloaded from cv2's github
# 3
facecascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


recognizer = cv.face.LBPHFaceRecognizer_create() 
recognizer.read('t.yml')

# converts into a video
# 2
while True:
    # updates the frame continuously
    ret,frame = vid.read()
    img = cv.flip(frame,1)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    
   
    f = facecascade.detectMultiScale(grey)
   
    # detectMultiScale returns a tuple containing an x, y, h, and w value of the detected face
    # the parameters of the funvctio
    

    for(x,y,h,w) in f:
        roi = cv.rectangle(grey, (x, y), (x+w, y+h), (255, 182, 193), 3)

    
    # for(x,y,h,w) in s:
    #     cv.rectangle(frame, (x, y), (x+w, y+h), (25, 0, 93), 3)

    id , conf = recognizer.predict(grey)




    if conf > 45:
        print(conf)
        print(reversed_label[id])
        # i tried conf above 35 and 65 but 35 wasnt accurate at all and 65 outputed "uknown face"

    else:
        print("unknown face")
    # shows the camera
    cv.imshow("camera", grey)
    
    # closes the cam
    if cv.waitKey(1) == ord("c") or cv.waitKey(1) == ord("C"):
        break

    
# for some reason the AI detects cameron diaz and marcia gray harden, sometimes. 

    

    
    
