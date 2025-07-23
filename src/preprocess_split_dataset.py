import os
import cv2
import numpy as np

IMG_SIZE = 299  # For Xception input
CLASSES = ["Normal", "TB_Positive"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_DATASET_DIR = os.path.join(BASE_DIR, "../split_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "../model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(split):
    data = []
    labels = []
    print(f"\n🔄 Processing {split} set...")
    for label_name in CLASSES:
        class_path = os.path.join(SPLIT_DATASET_DIR, split, label_name)
        label = 0 if label_name == "Normal" else 1

        if not os.path.exists(class_path):
            print(f"❌ Missing: {class_path}")
            continue

        for file in os.listdir(class_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Could not read {img_path}")
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)  # Shape: (299, 299, 1)
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"⚠️ Error with {img_path}: {e}")

    return np.array(data), np.array(labels)

# Process all splits
splits = {}
for split in ["train", "val", "test"]:
    X, y = preprocess(split)
    splits[split] = (X, y)
    np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), y)
    print(f"✅ Saved {split} set — X: {X.shape}, y: {y.shape}")

print("\n🎉 All datasets preprocessed and saved!")
