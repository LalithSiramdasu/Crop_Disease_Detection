import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = "Dataset/PlantVillage"
OUTPUT_DIR = "Dataset/split_data"

# Split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Create output folders
for split in ["train", "valid", "test"]:
    split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# Loop through each category (class folder)
for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)

    # Shuffle images
    random.shuffle(images)

    # Split data
    train_images, temp = train_test_split(images, test_size=(1 - TRAIN_RATIO), random_state=42)
    valid_images, test_images = train_test_split(temp, test_size=(TEST_RATIO / (TEST_RATIO + VALID_RATIO)), random_state=42)

    # Function to copy images
    def copy_images(img_list, split):
        split_category_path = os.path.join(OUTPUT_DIR, split, category)
        os.makedirs(split_category_path, exist_ok=True)
        for img in img_list:
            src = os.path.join(category_path, img)
            dst = os.path.join(split_category_path, img)
            shutil.copy(src, dst)

    # Copy files
    copy_images(train_images, "train")
    copy_images(valid_images, "valid")
    copy_images(test_images, "test")

print("✅ Dataset successfully split into Train, Validation, and Test sets!")
