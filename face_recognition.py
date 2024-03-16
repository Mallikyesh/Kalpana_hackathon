import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['madhusudan', 'mahanta', 'mallikarjun']

# features=np.load('features.npy')
# labels=np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

img=cv.imread(r'C:\Users\mahan\OneDrive\Pictures\opencv\mallikarjun\20240316_075103.jpg')

def rescaleFrame(frame, scale=0.3):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

frame_resized=rescaleFrame(img)

gray=cv.cvtColor(frame_resized,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+h]

label,confidence=face_recognizer.predict(faces_roi)
print(f'label={label} with confidence of {confidence} ')

cv.putText(frame_resized,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX, 1.0,(0,255,0),thickness=2)
cv.rectangle(frame_resized, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('detected face',frame_resized)

cv.waitKey(0)