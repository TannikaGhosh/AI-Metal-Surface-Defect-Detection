# AI-Metal-Surface-Defect-Detection

A locally deployed AI system for metal surface defect detection, inspired by techniques used in medical image analysis. The model leverages a Convolutional Neural Network (CNN) trained on the NEU dataset and custom image sources to classify surfaces as **GOOD** or **DEFECTED**. It includes a real-time Gradio interface for instant predictions with confidence scores, enabling efficient automated quality inspection.


## Setup
1. Create a virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`

## 🚀 How to Run
1. Unzip the NEU dataset into `neu_data/` and extract images from the PDFs using `python scripts/extract_pdf_images.py`.
2. Prepare the NEU training images using `python scripts/prepare_neu_data.py`.
3. Combine the good images into the training pool using `python scripts/build_training_folder.py`.
4. Train the model by running `python scripts/train_model.py`. The model will be saved to the `models/` directory.
5. Launch the dashboard by running:
   ```bash
   cd dashboard
   python app.py
   ```
   Access the live dashboard at `http://localhost:7860`.
## Dataset
The dataset is too large to host on GitHub.  
Download it from Google Drive: [Click here](https://drive.google.com/drive/folders/1VfAAOcaBNyV3ztp4M6bv9VUIqvuGLV8A?usp=drive_link)
