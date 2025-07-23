import os

# Set your dataset directory path
base_dir = "dataset"

# Define rename rules
rename_rules = {
    "Normal": "n",
    "TB_Positive": "tb"
}

for class_name, prefix in rename_rules.items():
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"❌ Folder not found: {class_dir}")
        continue

    files = sorted(os.listdir(class_dir))  # Sort to maintain order
    count = 1

    for file in files:
        file_path = os.path.join(class_dir, file)
        if not os.path.isfile(file_path):
            continue

        # Get extension
        ext = os.path.splitext(file)[-1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            continue

        new_name = f"{prefix}{count}{ext}"
        new_path = os.path.join(class_dir, new_name)

        try:
            os.rename(file_path, new_path)
            count += 1
        except Exception as e:
            print(f"⚠️ Failed to rename {file_path}: {e}")

print("✅ All files renamed successfully.")
