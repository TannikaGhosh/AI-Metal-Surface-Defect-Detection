# scripts/test_batch.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model("models/defect_detection_model.h5") # Fixed model path

def test_image(img_path):
    # Ensure image size matches the trained model (128x128 we used instead of 150x150)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    # Classes map explicitly: 'good' is 0, 'defected' is 1
    label = "DEFECTED" if pred > 0.5 else "GOOD"
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"{img_path}: {label} ({confidence:.2f})")

# Test all good testing images
good_folder = "extracted_images/good_for_testing"
if os.path.exists(good_folder):
    for f in os.listdir(good_folder):
        test_image(os.path.join(good_folder, f))

# Test all defected PDF images
defect_folder = "extracted_images/defected_from_pdf_for_testing"
if os.path.exists(defect_folder):
    for f in os.listdir(defect_folder):
        test_image(os.path.join(defect_folder, f))
