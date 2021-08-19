import os
import cv2 as cv
import numpy as np

# STEP 1 : Selecting Data For Modeling
# ____________________________________________________________________________


# Read Cascade Classifier from haar_face.xml
haar_cascade = cv.CascadeClassifier('../haar_face.xml')
# Location of Training dataset
DIR = './train'

features = []  # Features for training (faces of people)
labels = []  # For labels corresponding to features (whose face is it)
people = []  # List of people in dataset
for folder in os.listdir(DIR):
    people.append(folder)
print(f'List of people : {people}')


def create_data():
    """
    Creates features and labels
    :return: None
    """
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            path_img = os.path.join(path, img)
            get_img = cv.imread(path_img)
            grey = cv.cvtColor(get_img, cv.COLOR_BGR2GRAY)
            # Detecting faces in image(numpy array) 'grey' in rectangular co-ordinate system
            faces_rect = haar_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = grey[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_data()
print("________features and labels creation successful________")


# STEP 2 : Creating, Training and Saving our model
# ____________________________________________________________________________


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)
print("________LBPH Face Recognizer training successful________")

face_recognizer.save('trained_face_model.yml')
print('________Saving Trained model in "trained_face_model.yml"________')

