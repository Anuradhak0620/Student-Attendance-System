from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import datetime
import time

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Path to the known faces folder (update this to your folder's correct path)
known_faces_folder = 'known_faces/Anuradha'

# Get all the image paths from the folder
image_paths = [os.path.join(known_faces_folder, f) for f in os.listdir(known_faces_folder) if f.endswith('.jpg')]

# Check if there are images in the folder
if len(image_paths) == 0:
    raise FileNotFoundError(f"No images found in {known_faces_folder}")

# Load the images and convert them to grayscale
known_faces = []
labels = []

# Load and prepare training data (use only one image for simplicity)
for i, image_path in enumerate(image_paths):
    face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if face is not None:
        known_faces.append(face)
        labels.append(i)  # Use the index as label (e.g., 0, 1, 2, 3 for each photo)
    else:
        print(f"Warning: Unable to load image {image_path}")

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.train(known_faces, np.array(labels))  # Train with the images and labels

# Define a variable to track when attendance was last logged
last_logged_time = time.time()
last_recognized_label = None  # Track the last recognized label

@app.route('/')
def index():
    return render_template('index.html')  # Show the webcam feed in the browser

def generate_frames():
    global last_logged_time, last_recognized_label
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Load OpenCV face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Loop through each face detected in the frame
            for (x, y, w, h) in faces:
                # Extract the face from the frame
                face = gray[y:y+h, x:x+w]

                # Recognize the face
                label, confidence = recognizer.predict(face)

                # If the confidence is below a threshold, recognize the face
                if confidence < 70:  # Reduced the threshold for better recognition
                    name = "Anuradha"  # Recognized face
                    color = (0, 255, 0)  # Green color for recognized face

                    # Only log if the recognized label changes (to avoid redundant logs)
                    if label != last_recognized_label:
                        # Log the attendance with time
                        current_time = time.time()
                        if current_time - last_logged_time >= 10:  # 10 seconds threshold
                            try:
                                with open('attendance_log.txt', 'a') as f:
                                    f.write(f"{name} recognized at {datetime.datetime.now()}\n")
                                    print(f"Logged attendance for {name}")  # Debugging log
                                last_logged_time = current_time  # Update the last logged time
                                last_recognized_label = label  # Update the last recognized label
                            except Exception as e:
                                print(f"Error writing to log: {e}")  # Debugging log
                else:
                    name = "Unknown"  # If not recognized
                    color = (0, 0, 255)  # Red color for unknown face
                    last_recognized_label = None  # Reset if face is not recognized

                # Draw rectangle around face and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in HTTP format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
