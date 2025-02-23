from tensorflow.keras.models import load_model

# Load the model from the 'models' folder
model = load_model("models/drowsiness_detection_model.h5")
print("Model loaded successfully.")
