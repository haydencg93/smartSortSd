import os
import zipfile
import shutil
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

print("Downloading dataset...")
file_path = hf_hub_download(repo_id="garythung/trashnet", filename="dataset-resized.zip", repo_type="dataset")

print("Extracting files...")
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall("raw_data")

base_dir = "data_6class"
original_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for split in ['train', 'val']:
    for c in original_classes:
        os.makedirs(os.path.join(base_dir, split, c), exist_ok=True)

source_dir = "raw_data/dataset-resized"
for c in original_classes:
    class_dir = os.path.join(source_dir, c)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        
        # Robust 80/20 split using scikit-learn
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        for img in train_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, 'train', c, img))
            
        for img in val_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, 'val', c, img))

print("Dataset successfully organized into 6 base classes with a standardized split!")