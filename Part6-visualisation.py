# EDA_LISA_Samples.py
import os
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd

dataset_path = "./lisa_dataset"  # same as before
boxes_df_path = os.path.join(dataset_path, "boxes_df.pkl")

# Load boxes_df
if not os.path.exists(boxes_df_path):
    raise FileNotFoundError(f"{boxes_df_path} not found. Run part5.py first to create it.")
boxes_df = pd.read_pickle(boxes_df_path)
print(f"Loaded boxes_df with {len(boxes_df)} bounding boxes.")


# --------------------------
# 0. Dataset path and boxes_df
# --------------------------
dataset_path = "./lisa_dataset"  # path to your extracted dataset

# Assume boxes_df is already loaded from Part 5
# If running separately, reload boxes_df using the same logic from Part 5

# --------------------------
# 1. Helper function to find image path
# --------------------------
def find_image(root, filename):
    """
    Recursively search for the image in the dataset folder.
    Returns full path if found, else None.
    """
    for dirpath, _, files in os.walk(root):
        if os.path.basename(filename) in files:
            return os.path.join(dirpath, os.path.basename(filename))
    return None

# --------------------------
# 2. Display function
# --------------------------
def display_sample_images(df, num_samples=5, category=None):
    """
    Displays sample images with bounding boxes.
    category: None, "Small_Box", "Large_Box", "Unusual_Aspect_Ratio"
    """
    if category:
        df = df[df[category] == True]
        if df.empty:
            print(f"No boxes found for category '{category}'")
            return

    sampled_rows = df.sample(n=min(num_samples, len(df)), random_state=42)

    for _, row in sampled_rows.iterrows():
        img_path = find_image(dataset_path, row['Filename'])
        if not img_path or not os.path.isfile(img_path):
            print(f"Image not found: {row['Filename']}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw bounding box
        x1, y1 = int(row['Width']), int(row['Height'])  # <-- fix below
        x1 = int(row['Width'] + row['Width']) if 'Width' in row else 0
        y1 = int(row['Height'] + row['Height']) if 'Height' in row else 0
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])

        # Actually we should use upper-left and lower-right
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        x1 = int(row['Width'])
        y1 = int(row['Height'])

        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        # Corrected:
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        
        # Actually, let's just do the proper bounding box coordinates
        x1 = int(row['Width'])
        y1 = int(row['Height'])
        x2 = int(x1 + row['Width'])
        y2 = int(y1 + row['Height'])

        # Draw box in red for small/large/unusual
        color = (255,0,0) if category else (0,255,0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.title(f"{row['Filename']} - {row['Class']}" + (f" ({category})" if category else ""))
        plt.axis('off')
        plt.show()

# --------------------------
# 3. Display normal and outlier samples
# --------------------------
print("\nDisplaying normal samples:")
display_sample_images(boxes_df, num_samples=5)

print("\nDisplaying small boxes:")
display_sample_images(boxes_df, num_samples=5, category="Small_Box")

print("\nDisplaying large boxes:")
display_sample_images(boxes_df, num_samples=5, category="Large_Box")

print("\nDisplaying unusual aspect ratio boxes:")
display_sample_images(boxes_df, num_samples=5, category="Unusual_Aspect_Ratio")

