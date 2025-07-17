import os
import face_recognition
from sklearn import svm
import pickle
import numpy as np

# Path to your dataset
dataset_path = './Dataset'

# List to store all face encodings and labels
encodings = []
names = []

# Loop through each person's folder in the dataset
print("Loading and encoding faces...")
for person_folder in os.listdir(dataset_path):
    person_folder_path = os.path.join(dataset_path, person_folder)

    # Skip if it's not a directory
    if not os.path.isdir(person_folder_path):
        print(f"Skipping {person_folder_path}, not a directory.")
        continue

    # Extract the person's name from the folder name
    person_name = person_folder

    # Loop through each image in the person's folder
    for img_file in os.listdir(person_folder_path):
        img_path = os.path.join(person_folder_path, img_file)

        # Skip if it's not a file
        if not os.path.isfile(img_path):
            print(f"Skipping {img_path}, not a file.")
            continue

        print(f"Processing image: {img_path}")
        try:
            image = face_recognition.load_image_file(img_path)
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            continue

        # Get face encodings
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            encodings.append(face_encodings[0])
            names.append(person_name)
            print(f"Encoded {img_file} as {person_name}")
        else:
            print(f"No face found in {img_file}")

# Ensure we have encodings to train on
if not encodings:
    print("No face encodings found. Please check your dataset and try again.")
    exit(1)

# Train a SVM classifier
print("Training the SVM classifier...")
clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings, names)

# Save the trained classifier
model_path = 'Facial_Recognition.sav'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"Model saved to {model_path}")
