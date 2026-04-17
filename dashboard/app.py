import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model
model_path = os.path.join("..", "models", "defect_detection_model.h5")
model = tf.keras.models.load_model(model_path)

def predict_image(img):
    # img is a PIL image
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    # Classes map explicitly: 'good' is 0, 'defected' is 1
    if prediction > 0.5:
        return f"🔴 **DEFECTED** (confidence: {prediction:.2f})"
    else:
        return f"🟢 **GOOD** (confidence: {1 - prediction:.2f})"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🔩 Metal Surface Defect Detector",
    description="Upload an image of a metal surface. The AI will classify it as **GOOD** or **DEFECTED**.\n\nTrained on NEU dataset (defects) + good images extracted from your PDF.",
    examples=[]
)

# Launch with a public link (share=True)
iface.launch(share=True)
