import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dirs = [
    os.path.join(BASE_DIR, "../dataset/Montgomery"),
    os.path.join(BASE_DIR, "../dataset/Shenzhen")
]

classes = ["Normal", "TB_Positive"]

for dataset_dir in dataset_dirs:
    print(f"📁 Dataset: {os.path.basename(dataset_dir)}")
    for cls in classes:
        class_path = os.path.join(dataset_dir, cls)
        if not os.path.exists(class_path):
            print(f"❌ Folder not found: {class_path}")
            continue
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  {cls}: {count} images")