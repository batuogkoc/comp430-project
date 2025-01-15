import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "unsplit"
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

image_files = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            image_files.append(os.path.join(root, file))

print(f"Total images found: {len(image_files)}")

train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)  # 0.33 of 30% is 10%

def copy_files(file_list, target_dir):
    for file in file_list:
        filename = os.path.basename(file)
        shutil.copy(file, os.path.join(target_dir, filename))

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print(f"Training images: {len(train_files)}, Validation images: {len(val_files)}, Testing images: {len(test_files)}")
