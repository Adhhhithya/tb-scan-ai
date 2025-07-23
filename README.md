# 🩻 TB-Scan-AI: Tuberculosis Detection from Chest X-ray Images

A deep learning-based diagnostic tool to detect **Tuberculosis (TB)** from chest X-ray images using **Xception** transfer learning and real-world datasets. This project focuses on robust medical image classification with scalable training using **Azure GPU VM**.

---

## 🎯 Project Goals

- Accurately detect TB in chest X-rays *(binary classification: TB vs Normal)*
- Use real-world open-access datasets (Montgomery + Shenzhen)
- Train an advanced model (**Xception**) for high generalization
- Enable training on **Azure cloud** and local testing in **VS Code**
- Export a deployable `.h5` model for prediction

---

## ✅ Current Status

| Stage            | Description |
|------------------|-------------|
| ✅ Data Prep      | Combined and preprocessed Montgomery + Shenzhen datasets |
| ✅ Preprocessing  | Images resized to `299x299x3`, normalized, and saved as `.npy` |
| ✅ Splitting      | Data split into `train/val/test` sets using stratification |
| ✅ Model          | **Xception** with frozen base + dense head trained on Azure VM (CPU fallback) |
| ✅ Accuracy       | **Training acc: 99.1%**, **Validation acc: 93.3%**, **Test acc: 95.0%** |
| ✅ Evaluation     | Confusion matrix, classification report generated |
| ✅ Export         | Model saved as `model/xception_tb_model.h5` |
| 🔜 Next           | Real X-ray testing and optional deployment via web/app |

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras (Xception)
- NumPy, Matplotlib, Scikit-learn
- Azure VM (CPU/GPU)
- VS Code + SSH/Remote Sync

---

## 🗂️ Project Structure

```bash
TB-Scan-AI/
│
├── data/                      # Datasets (.npy files)
│   ├── X_xception.npy         # Preprocessed image data (299x299x3)
│   └── y_xception.npy         # Labels (0 = Normal, 1 = TB)
│
├── model/                    
│   └── xception_tb_model.h5   # Trained model
│
├── scripts/
│   ├── preprocess_dataset.py  # Resize & normalize raw images
│   ├── split_dataset.py       # Stratified train/val/test split
│   └── train_xception.py      # Training script (for Azure/local)
│
├── test/                      # Test image scripts (upcoming)
│   └── predict_on_image.py    # (Planned) Predict TB from input image
│
├── venv/                      # Python virtual environment (optional)
├── requirements.txt           # Python dependencies
└── README.md
