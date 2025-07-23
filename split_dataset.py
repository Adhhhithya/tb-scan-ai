import os
import shutil
import random

# Configuration
SOURCE_DIR = "dataset"  # Your merged dataset with 'Normal' and 'TB_Positive' folders
TARGET_DIR = "split_dataset"  # Output directory for split data
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}
CLASSES = ["Normal", "TB_Positive"]

def ensure_dirs():
    for split in SPLIT_RATIOS.keys():
        for cls in CLASSES:
            split_path = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(split_path, exist_ok=True)

def split_and_copy():
    for cls in CLASSES:
        src_folder = os.path.join(SOURCE_DIR, cls)
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIOS["train"] * total)
        val_end = train_end + int(SPLIT_RATIOS["val"] * total)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in split_map.items():
            for img_name in files:
                src_path = os.path.join(src_folder, img_name)
                dest_path = os.path.join(TARGET_DIR, split, cls, img_name)
                shutil.copy2(src_path, dest_path)

        print(f"✅ {cls}: {total} images split into Train: {train_end}, Val: {val_end - train_end}, Test: {total - val_end}")

if __name__ == "__main__":
    random.seed(42)
    ensure_dirs()
    split_and_copy()
    print("🎉 Dataset split completed!")
