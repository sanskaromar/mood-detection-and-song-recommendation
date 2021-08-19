import cv2 as cv
import os

# STEP 3 : Making Predictions using our trained model
# ____________________________________________________________________________


# Read Cascade Classifier from haar_face.xml
haar_cascade = cv.CascadeClassifier('../haar_face.xml')
# Location of Training dataset
DIR = './train'

people = []  # List of people in dataset
for folder in os.listdir(DIR):
    people.append(folder)

# Instantiate and then read model from 'trained_face_model.yml'
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('./trained_face_model.yml')

# Reading any image from test data
img = cv.imread(r'./val/jerry_seinfeld/2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detecting faces in image(numpy array) 'grey' in rectangular co-ordinate system
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
(x, y, w, h) = faces_rect[0]
faces_roi = gray[y:y+h, x:x+w]

# Making Predictions
label, confidence = face_recognizer.predict(faces_roi)
print(f'label: {people[label]}  confidence: {confidence}')

# Displaying Detected face with model's prediction
cv.putText(img, str(people[label]), (x, y+h+11), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=2)
cv.putText(img, str(round(confidence, 2)) + "%", (x, y-2), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=2)
cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
