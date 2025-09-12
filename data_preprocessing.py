import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
BASE_DIR = "Dataset/split_data"
train_dir = os.path.join(BASE_DIR, "train")
valid_dir = os.path.join(BASE_DIR, "valid")
test_dir = os.path.join(BASE_DIR, "test")

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # Normalize pixel values (0–1)
    rotation_range=20,        # Random rotation
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    shear_range=0.2,          # Shearing
    zoom_range=0.2,           # Zoom
    horizontal_flip=True,     # Flip horizontally
    fill_mode='nearest'       # Fill missing pixels
)

# Validation/Test should not be augmented (only rescaled)
valid_test_datagen = ImageDataGenerator(rescale=1.0/255)

# Create Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = valid_test_datagen.flow_from_directory(
    valid_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = valid_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Print class indices
print("✅ Classes:", train_generator.class_indices)
