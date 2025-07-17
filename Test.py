# TESTING

import cv2
import face_recognition
import pickle
import numpy as np
# Load the trained classifier
model_path = 'Facial_Recognition.sav'
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        probabilities = clf.predict_proba([face_encoding])[0]
        best_match_index = np.argmax(probabilities)
        name = "Unknown"
        confidence = probabilities[best_match_index]

        if confidence > 0.6:
            name = clf.classes_[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name and confidence below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = f"{name}: {confidence:.2f}"
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
