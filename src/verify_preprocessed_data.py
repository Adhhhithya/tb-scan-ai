import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed data
with open("../model/preprocessed_data.pkl", "rb") as f:
    data, labels = pickle.load(f)

# Basic checks
print(f"✅ Data shape: {data.shape}")          # Expecting (num_samples, 224, 224, 1)
print(f"✅ Labels shape: {labels.shape}")      # Expecting (num_samples,)
print(f"✅ Unique labels: {np.unique(labels)}")  # Should be [0 1]

# Check a sample image
index = 0  # you can change this to 1, 2, etc.
img = data[index]
label = labels[index]

print(f"Sample image shape: {img.shape}")
print(f"Label (0 = Normal, 1 = TB): {label}")
print(f"Pixel value range: min={img.min()}, max={img.max()}")

# Plot the image
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Label: {'TB_Positive' if label == 1 else 'Normal'}")
plt.axis('off')
plt.show()
