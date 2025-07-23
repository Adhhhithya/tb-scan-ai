import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")  # <-- FIXED

IMG_SIZE = 299  # For Xception input

def load_dataset(dataset_dir):
    data = []
    labels = []
    for label_name in ["Normal", "TB_Positive"]:
        class_path = os.path.join(dataset_dir, label_name)
        label = 0 if label_name == "Normal" else 1

        if not os.path.exists(class_path):
            print(f"❌ Folder not found: {class_path}")
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
                img = np.expand_dims(img, axis=-1)
                img = img / 255.0  # Normalize

                data.append(img)
                labels.append(label)

            except Exception as e:
                print(f"⚠️ Skipping {img_path} — {e}")

    return np.array(data), np.array(labels)

# Load data
print("🔄 Loading merged dataset...")
X, y = load_dataset(DATASET_DIR)

print("✅ All data loaded")
print(f" Data shape: {X.shape}")         # (num_samples, 299, 299, 1)
print(f" Labels shape: {y.shape}")       # (num_samples,)
print(f" Unique labels: {np.unique(y)}")
if X.shape[0] > 0:
    print(f" Sample image shape: {X[0].shape}")
    print(f" Pixel value range: min={X.min()}, max={X.max()}")
else:
    print("⚠️ No images loaded. Check your dataset paths and folder names.")

# Save
if X.shape[0] > 0:
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
    np.save(os.path.join(BASE_DIR, "model/X_xception.npy"), X)
    np.save(os.path.join(BASE_DIR, "model/y_xception.npy"), y)
    print("💾 Saved as X_xception.npy and y_xception.npy")
else:
    print("❌ Nothing saved. No data found.")
