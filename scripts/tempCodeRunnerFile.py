import cv2
import numpy as np
import tensorflow as tf
import time
from playsound import playsound  # Ensure you have installed this with: pip install playsound

# Use the correct image dimensions expected by the model (224x224)
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Load Haar cascade for face detection (grayscale is used for face detection)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load the trained drowsiness detection model
model = tf.keras.models.load_model("models/drowsiness_detection_model.h5")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened successfully. Starting real-time detection...")

# Timer to avoid playing alarm sound continuously
last_alert_time = time.time()

# Adjusted threshold for drowsiness detection
THRESHOLD = 0.6  # Adjust based on actual prediction values

while True:
    try:
        ret, frame = cap.read()
        
        # If webcam frame is not captured, retry
        if not ret:
            print("Warning: Frame capture failed. Reopening webcam...")
            cap.release()
            cap = cv2.VideoCapture(0)  # Reinitialize webcam
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]  # Crop the face region

            # Resize the face image to match model input size
            input_img = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

            # Normalize if required by the model
            input_img = (input_img - 127.5) / 127.5  # Change normalization if needed
            input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension

            # Predict drowsiness
            prediction = model.predict(input_img)[0][0]
            print(f"Model Prediction: {prediction}")  # Debugging output

            # Draw rectangle around detected face
            color = (0, 255, 0) if prediction >= THRESHOLD else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display prediction text
            text = "Awake" if prediction >= THRESHOLD else "Drowsy"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Play an alert sound if the person is drowsy
            if text == "Drowsy":
                current_time = time.time()
                if current_time - last_alert_time > 5:  # Play sound only every 5 seconds
                    playsound("alarm.wav")
                    last_alert_time = current_time

        # Show the output frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        # Ensure the program doesn't exit unexpectedly
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        continue  # Keep running even if an error occurs

# Release resources
cap.release()
cv2.destroyAllWindows()