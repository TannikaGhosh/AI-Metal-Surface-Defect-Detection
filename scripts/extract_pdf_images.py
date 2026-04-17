# scripts/extract_pdf_images.py
import fitz  # PyMuPDF
import os
import random
from PIL import Image
from io import BytesIO

def extract_images_from_pdf(pdf_path, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for page_num in range(len(doc)):
        images = doc.get_page_images(page_num)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_filename = f"{prefix}_{count+1:04d}.png"
            image.save(os.path.join(output_folder, img_filename))
            count += 1
    print(f"Extracted {count} images from {pdf_path}")
    return count

# Paths
good_pdf = "metal_defect_detection/good_image.pdf"
defected_pdf = "metal_defect_detection/defected_image.pdf"

# 1. Extract all good images temporarily
temp_good_folder = "extracted_images/temp_good_all"
extract_images_from_pdf(good_pdf, temp_good_folder, "good")

# 2. Split good images into training (70%) and testing (30%)
all_good = os.listdir(temp_good_folder)
random.shuffle(all_good)
split_idx = int(0.7 * len(all_good))
train_good = all_good[:split_idx]
test_good = all_good[split_idx:]

train_good_folder = "extracted_images/good_for_training"
test_good_folder = "extracted_images/good_for_testing"
os.makedirs(train_good_folder, exist_ok=True)
os.makedirs(test_good_folder, exist_ok=True)

for fname in train_good:
    os.rename(os.path.join(temp_good_folder, fname), os.path.join(train_good_folder, fname))
for fname in test_good:
    os.rename(os.path.join(temp_good_folder, fname), os.path.join(test_good_folder, fname))
os.rmdir(temp_good_folder)

print(f"Good images: {len(train_good)} for training, {len(test_good)} for testing")

# 3. Extract defected images from defected.pdf (all for testing)
defected_test_folder = "extracted_images/defected_from_pdf_for_testing"
extract_images_from_pdf(defected_pdf, defected_test_folder, "defect")