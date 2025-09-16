import cv2
import os
import numpy as np

# Path to training images
data_path = 'known_faces'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
labels = []
label_id = 0
label_dict = {}

for person_name in os.listdir(data_path):
    person_path = os.path.join(data_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_dict[label_id] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in face_rects:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))  # Resize to same size
            faces.append(face)
            labels.append(label_id)
            break  # Only one face per image

    label_id += 1

# Train and save the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save('trainer.yml')

print("Training complete. Model saved as trainer.yml.")

