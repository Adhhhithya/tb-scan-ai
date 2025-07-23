import cv2
import numpy as np
import tensorflow as tf
import os

# Load model
model = tf.keras.models.load_model("../model/tb_cnn_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Error: Could not read image '{image_path}'.")
        return

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 1)

    prediction = model.predict(img, verbose=0)[0][0]
    label = "TB Detected ✅" if prediction > 0.5 else "Normal 🫁"
    print(f"Prediction for '{os.path.basename(image_path)}': {label} (Confidence: {prediction:.2f})")

if __name__ == "__main__":
    test_folder = "../dataset/tb-Positive/"  # Change to "Normal" if needed
    files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for file in files[:5]:  # Test only first 5 images
        image_path = os.path.join(test_folder, file)
        predict_image(image_path)
