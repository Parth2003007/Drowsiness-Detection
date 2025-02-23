import cv2
import numpy as np
import tensorflow as tf
import time
from playsound import playsound  # Ensure you have installed this with: pip install playsound

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load trained drowsiness detection model
model = tf.keras.models.load_model("models/drowsiness_detection_model.h5")

# Open the webcam
cap = cv2.VideoCapture(0)

# Timer to avoid playing alarm sound continuously
last_alert_time = time.time()

# Threshold for classifying drowsiness (adjust as needed)
THRESHOLD = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with slightly adjusted parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) == 0:
        print("[INFO] No face detected.")

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]

        # Resize and normalize the face image for the model
        face_resized = cv2.resize(roi_color, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        input_img = np.expand_dims(face_normalized, axis=0)

        # Get prediction from the model
        prediction = model.predict(input_img)[0][0]
        print(f"[DEBUG] Face at ({x}, {y}, {w}, {h}) -> Prediction: {prediction:.4f}")

        # Decide based on prediction value
        if prediction > THRESHOLD:
            cv2.putText(frame, "DROWSY!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Play alarm if more than 5 seconds have passed since last alert
            if time.time() - last_alert_time > 5:
                playsound("alarm.wav")  # Make sure "alarm.wav" is in your project folder
                last_alert_time = time.time()
        else:
            cv2.putText(frame, "AWAKE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()