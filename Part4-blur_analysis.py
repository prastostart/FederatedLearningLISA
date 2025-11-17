# EDA_LISA_Blur.py

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 0. Dataset path
# --------------------------
dataset_path = "./lisa_dataset"  # path to your extracted dataset

# --------------------------
# 1. Collect all images
# --------------------------
image_files = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
print(f"Found {len(image_files)} images in dataset.")

# --------------------------
# 2. Compute blur score using Laplacian variance
# --------------------------
def compute_blur(img_path):
    """
    Compute blur score for an image using variance of Laplacian
    Higher variance = sharper image
    Lower variance = blurrier image
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

blur_scores = []

for img_file in image_files:
    score = compute_blur(img_file)
    if score is not None:
        blur_scores.append(score)

blur_scores = np.array(blur_scores)
print(f"Computed blur scores for {len(blur_scores)} images.")

# --------------------------
# 3. Plot histogram of blur scores
# --------------------------
plt.figure(figsize=(8,5))
plt.hist(blur_scores, bins=50, color='lightgreen', edgecolor='black')
plt.title("Blur Score Distribution Across Dataset")
plt.xlabel("Laplacian Variance (Blur Score)")
plt.ylabel("Number of Images")
plt.show()

# --------------------------
# 4. Optional: Day vs Night
# --------------------------
day_folders = ['dayTrain', 'daySequence1', 'daySequence2', 'sample-dayClip6']
night_folders = ['nightTrain', 'nightSequence1', 'nightSequence2', 'sample-nightClip1']

day_blur = []
night_blur = []

for img_file in image_files:
    relative_path = os.path.relpath(img_file, dataset_path)
    folder = relative_path.split(os.sep)[0]
    score = compute_blur(img_file)
    if score is None:
        continue
    if folder in day_folders:
        day_blur.append(score)
    elif folder in night_folders:
        night_blur.append(score)

plt.figure(figsize=(8,5))
plt.hist(day_blur, bins=50, alpha=0.6, label='Day', color='gold')
plt.hist(night_blur, bins=50, alpha=0.6, label='Night', color='navy')
plt.title("Blur Score Distribution: Day vs Night")
plt.xlabel("Laplacian Variance (Blur Score)")
plt.ylabel("Number of Images")
plt.legend()
plt.show()

# --------------------------
# 5. Summary statistics
# --------------------------
print("Blur Score Summary:")
print(f"Overall - mean: {blur_scores.mean():.2f}, std: {blur_scores.std():.2f}")
print(f"Day - mean: {np.mean(day_blur):.2f}, std: {np.std(day_blur):.2f}")
print(f"Night - mean: {np.mean(night_blur):.2f}, std: {np.std(night_blur):.2f}")

# --------------------------
# 6. Optional: Flag very blurry images
# --------------------------
threshold = 100  # tweak as needed
very_blurry = [f for f, s in zip(image_files, blur_scores) if s < threshold]
print(f"\nNumber of very blurry images (score < {threshold}): {len(very_blurry)}")
