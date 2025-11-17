# EDA_LISA_Brightness.py

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# 0. Dataset path
# --------------------------
dataset_path = "./lisa_dataset"  # path to your extracted dataset

# --------------------------
# 1. Collect all images
# --------------------------
# Search recursively for all JPG images in dataset subfolders
image_files = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
print(f"Found {len(image_files)} images in dataset.")

# --------------------------
# 2. Compute brightness
# --------------------------
def compute_brightness(img_path):
    """
    Compute the average brightness of an image using the HSV V channel
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # brightness channel
    return np.mean(v_channel)

brightness_values = []

for img_file in image_files:
    b = compute_brightness(img_file)
    if b is not None:
        brightness_values.append(b)

brightness_values = np.array(brightness_values)
print(f"Computed brightness for {len(brightness_values)} images.")

# --------------------------
# 3. Plot brightness histogram
# --------------------------
plt.figure(figsize=(8,5))
plt.hist(brightness_values, bins=50, color='orange', edgecolor='black')
plt.title("Brightness Distribution Across Dataset")
plt.xlabel("Brightness (0-255)")
plt.ylabel("Number of Images")
plt.show()

# --------------------------
# 4. Optional: Split day vs night if folder names indicate
# --------------------------
day_folders = ['dayTrain', 'daySequence1', 'daySequence2', 'sample-dayClip6']
night_folders = ['nightTrain', 'nightSequence1', 'nightSequence2', 'sample-nightClip1']

day_brightness = []
night_brightness = []

for img_file in image_files:
    # Determine folder type
    relative_path = os.path.relpath(img_file, dataset_path)
    folder = relative_path.split(os.sep)[0]
    b = compute_brightness(img_file)
    if b is None:
        continue
    if folder in day_folders:
        day_brightness.append(b)
    elif folder in night_folders:
        night_brightness.append(b)

# Plot day vs night brightness
plt.figure(figsize=(8,5))
plt.hist(day_brightness, bins=50, alpha=0.6, label='Day', color='gold')
plt.hist(night_brightness, bins=50, alpha=0.6, label='Night', color='navy')
plt.title("Brightness Distribution: Day vs Night")
plt.xlabel("Brightness (0-255)")
plt.ylabel("Number of Images")
plt.legend()
plt.show()

# --------------------------
# 5. Summary statistics
# --------------------------
print("Brightness Summary:")
print(f"Overall - mean: {brightness_values.mean():.2f}, std: {brightness_values.std():.2f}")
print(f"Day - mean: {np.mean(day_brightness):.2f}, std: {np.std(day_brightness):.2f}")
print(f"Night - mean: {np.mean(night_brightness):.2f}, std: {np.std(night_brightness):.2f}")
