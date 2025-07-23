import os
import cv2
import matplotlib.pyplot as plt

for label in ["tb-positive", "normal"]:
    folder = os.path.join("..", "dataset", label)
    example = os.listdir(folder)[0]
    img = cv2.imread(os.path.join(folder, example))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4,4))
    plt.imshow(img_rgb)
    plt.title(f"Label: {label}, File: {example}")
    plt.axis('off')
    plt.show()
