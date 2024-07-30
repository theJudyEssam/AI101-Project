import os
import numpy as np
from PIL import Image
import pickle as p
import cv2 as cv 

# 4 
directory_name = os.path.dirname(os.path.abspath(__file__))
# path.dirname is used to get the name of the directory
# "it means that this method returns the pathname to the path passed as a parameter to this function.""
imgg = os.path.join(directory_name, r'C:\Users\Judy\Downloads\archive\database')
# join f directory wa7da

# print("testtest")



id_num = 0
# starting id number
labelids = {}
# empty dictionary to store all the ids in one place

x_train = []
y_labels = []
# stores all the independent values that i will eventually use to train the model
# stores all the labels in one list

facecascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml") 
recognizer = cv.face.LBPHFaceRecognizer_create()
# the openCV recognizer, not the most accurate but it works 
#  https://docs.opencv.org/3.4/df/d25/classcv_1_1face_1_1LBPHFaceRecognizer.html 

# 5
for root, dir, files in os.walk(imgg):
    # "walks" through the files and returns the root, directory and files

    for f in files:
        # 6
        if f.endswith('.jpg'):
            pathname = os.path.join(root, f)
            l = os.path.basename(root).replace(' ','-').lower()
            # l stands for label
            
            if l in labelids:
                continue
            else:
                labelids[l] = id_num
                id_num += 1

            id = labelids[l]
            # dh hay2ba zay indentifier kdaa


            # print(l, pathname)
            # 7
            pil = Image.open(pathname).convert("L") 
            # greyscale
            arr = np.array(pil, 'uint8')


            f = facecascade.detectMultiScale(arr)
            # print(arr)

            for (x , y, w, h) in f:
                region_of_i = arr[y: y+h , x: x+w]
                x_train.append(region_of_i)
                y_labels.append(id)

            # print(x_train)
            # print(id)
            

with open ('j.pickle', 'wb') as f:
    p.dump(labelids, f )

recognizer.train(x_train,np.array(y_labels))

# note: recognizer isn't that accurate, needs to be more accurate by either a) implementing a different one b)using a better dataset c)use a pre-trained network 

recognizer.save('t.yml')
# trainer





# used this as a references: https://pub.towardsai.net/how-to-create-a-new-custom-dataset-from-images-9b95977964ab 
# https://www.geeksforgeeks.org/opencv-python-program-face-detection/ 

