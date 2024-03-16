import os
import cv2 as cv
import numpy as np

people = ['madhusudan', 'mahanta', 'mallikarjun']
DIR = r'C:\Users\mahan\OneDrive\Pictures\opencv'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                print(f'Error: Unable to read image {img_path}')
                continue

            # Resize the image to 0.1 times its original size
            scaled_img = cv.resize(img_array, (0,0), fx=0.3, fy=0.3)

            gray = cv.cvtColor(scaled_img, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training done")

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Training the recognizer
face_recognizer.train(features, labels)

# Path where the file should be saved
save_path = 'face_trained.yml'

# Attempt to save the file
try:
    face_recognizer.save(save_path)
    print(f"Trained model saved to: {save_path}")
except Exception as e:
    print(f"Error saving trained model: {e}")

# Save features and labels
np.save('features.npy', features)
np.save('labels.npy', labels)
