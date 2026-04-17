# scripts/prepare_neu_data.py
import zipfile
import os
import shutil

# Unzip NEU dataset
with zipfile.ZipFile("metal-surface-defect-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("neu_data")
print("NEU dataset unzipped into neu_data/")

# The NEU folder structure is usually NEU-DET/ with subfolders Cr, In, Pa, Ps, Rs, Sc
# We will copy all images from all defect classes into training_data/defected/
defected_train_folder = "training_data/defected"
os.makedirs(defected_train_folder, exist_ok=True)

# Adjust for actual structure: Magnetic-Tile-Defect has defect subfolders
neu_root = "neu_data/Magnetic-Tile-Defect"
if os.path.exists(neu_root):
    defect_classes = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free"]
    for cls in defect_classes:
        cls_path = os.path.join(neu_root, cls, "Imgs")  # Images are in Imgs subfolder
        if os.path.exists(cls_path):
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                    shutil.copy(
                        os.path.join(cls_path, fname),
                        os.path.join(defected_train_folder, f"{cls}_{fname}")
                    )
else:
    # Fallback to original logic
    neu_root = "neu_data/NEU-DET"   # adjust if different
    if not os.path.exists(neu_root):
        # try alternative name
        for folder in os.listdir("neu_data"):
            if "NEU" in folder:
                neu_root = os.path.join("neu_data", folder)
                break

    defect_classes = ["Cr", "In", "Pa", "Ps", "Rs", "Sc"]
    for cls in defect_classes:
        cls_path = os.path.join(neu_root, cls)
        if os.path.exists(cls_path):
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
                    shutil.copy(
                        os.path.join(cls_path, fname),
                        os.path.join(defected_train_folder, f"{cls}_{fname}")
                    )
print(f"Copied defected images from NEU to training_data/defected/")