# scripts/build_training_folder.py
import shutil
import os

good_src = "extracted_images/good_for_training"
good_dst = "training_data/good"
if os.path.exists(good_src):
    shutil.copytree(good_src, good_dst, dirs_exist_ok=True)
    print(f"Copied good training images to {good_dst}")
else:
    print("No good training images found – run extraction first.")