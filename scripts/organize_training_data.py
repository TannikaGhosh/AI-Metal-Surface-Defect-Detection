# scripts/organize_training_data.py
import os
import shutil

# Copy good images from extracted_images/good_for_training to training_data/good
good_train_folder = "training_data/good"
os.makedirs(good_train_folder, exist_ok=True)

source_good = "extracted_images/good_for_training"
if os.path.exists(source_good):
    for fname in os.listdir(source_good):
        if fname.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
            shutil.copy(os.path.join(source_good, fname), os.path.join(good_train_folder, fname))
    print(f"Copied good images to training_data/good/")

# Also, copy good_for_testing to a test folder if needed, but for now, training data is ready
print("Training data organization complete.")