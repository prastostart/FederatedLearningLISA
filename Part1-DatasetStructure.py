# EDA_LISA_TrafficLights.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------
# 1. Set dataset path
# --------------------------
dataset_path = "./lisa_dataset"  # Change if your folder is different

# --------------------------
# 2. Dataset Structure & Basic Info
# --------------------------
print("=== Dataset Structure & Basic Info ===\n")

# List subfolders
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print("Folders inside dataset:")
for f in subfolders:
    print(" -", f)

# Count images per folder
def count_images(folder):
    return len(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))

print("\nImage counts per folder:")
for sf in subfolders:
    img_count = count_images(os.path.join(dataset_path, sf))
    print(f"{sf}: {img_count} images")

# --------------------------
# 3. Load Annotation CSVs
# --------------------------
annotation_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
dfs = {}

print("\nLoading annotation files:")
for ann_file in annotation_files:
    df_name = os.path.basename(ann_file).replace(".csv", "")
    df = pd.read_csv(ann_file, header=None)  # read raw CSV
    df_fixed = df[0].str.split(";", expand=True)
    df_fixed.columns = [
        "Filename",
        "Annotation tag",
        "Upper left corner X",
        "Upper left corner Y",
        "Lower right corner X",
        "Lower right corner Y",
        "Origin file",
        "Origin frame number",
        "Origin track",
        "Origin track frame number"
    ]
    dfs[df_name] = df_fixed
    print(f"Loaded {df_name} with {len(df_fixed)} annotations")

# --------------------------
# 4. Check for missing images
# --------------------------
print("\nChecking for missing images...")
for name, df in dfs.items():
    missing_count = 0
    for fname in df['Filename'].tolist():
        img_path = os.path.join(dataset_path, fname)
        if not os.path.isfile(img_path):
            missing_count += 1
    print(f"{name}: {missing_count} missing images out of {len(df)} annotations")

# --------------------------
# 5. Class-Level Analysis
# --------------------------
print("\n=== Class-Level Analysis ===")
for name, df in dfs.items():
    print(f"\n{name}:")
    classes = df['Annotation tag'].tolist()
    class_counts = Counter(classes)
    total = sum(class_counts.values())
    for cls, count in class_counts.items():
        print(f" - {cls}: {count} ({count/total*100:.2f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(6,4))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title(f"Class Distribution - {name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()
