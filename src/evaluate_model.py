import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
with open("../model/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
# Unpack tuple
X, y = data

# Ensure X has channel dimension and is float32
if X.ndim == 3:
    X = X[..., np.newaxis]
X = X.astype(np.float32)
if X.max() > 1.0:
    X = X / 255.0

# Ensure y is 1D
y = np.array(y).flatten()

# Load model
model = tf.keras.models.load_model("../model/tb_cnn_model.h5")

# Predict
y_pred = model.predict(X)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y, y_pred_labels)
print("✅ Classification Report:\n", classification_report(y, y_pred_labels, target_names=["Normal", "TB"]))

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "TB"], yticklabels=["Normal", "TB"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
