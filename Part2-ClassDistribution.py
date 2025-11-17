# EDA_LISA_Part1_2_Aggregate.py

import os
import glob
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# --------------------------
# 0. Dataset path
# --------------------------
dataset_path = "./lisa_dataset"  # path to your extracted dataset

# --------------------------
# Folder mapping (CSV folder -> actual dataset folder)
# --------------------------
folder_mapping = {
    "dayTraining": "dayTrain",
    "nightTraining": "nightTrain",
    "daySequence1": "daySequence1",
    "daySequence2": "daySequence2",
    "nightSequence1": "nightSequence1",
    "nightSequence2": "nightSequence2",
    "sample-dayClip6": "sample-dayClip6",
    "sample-nightClip1": "sample-nightClip1"
}

# --------------------------
# Part 1: Dataset Structure & Basic Info
# --------------------------
print("=== Part 1: Dataset Structure & Basic Info ===\n")

subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print("Folders inside dataset:")
for f in subfolders:
    print(" -", f)

def count_images(folder):
    return len(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))

print("\nImage counts per folder:")
for sf in subfolders:
    img_count = count_images(os.path.join(dataset_path, sf))
    print(f"{sf}: {img_count} images")

# --------------------------
# Part 2: Load Annotations & Class-Level Analysis
# --------------------------
print("\n=== Part 2: Load Annotations & Class-Level Analysis ===")

# Search for all CSV files recursively
annotation_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
if len(annotation_files) == 0:
    print("No CSV annotation files found!")
    exit(1)

print(f"Found {len(annotation_files)} annotation CSV files:\n")
for f in annotation_files:
    print(" -", f)

dfs = {}

# Aggregate counters
all_box_counts = Counter()
all_bulb_counts = Counter()

for ann_file in annotation_files:
    df_name = os.path.basename(ann_file).replace(".csv", "")
    df = pd.read_csv(ann_file, header=None)
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
    print(f"\nLoaded {df_name} with {len(df_fixed)} annotations")

    # Check missing images
    missing_count = 0
    for fname in df_fixed['Filename']:
        folder_in_csv = fname.split("/")[0]
        file_name = os.path.basename(fname)
        actual_folder = folder_mapping.get(folder_in_csv, folder_in_csv)
        img_path = os.path.join(dataset_path, actual_folder, file_name)
        if not os.path.isfile(img_path):
            missing_count += 1
    print(f"{df_name}: {missing_count} missing images out of {len(df_fixed)} annotations")

    # Aggregate counts
    counts = Counter(df_fixed['Annotation tag'])
    if "BOX" in df_name.upper():
        all_box_counts.update(counts)
    elif "BULB" in df_name.upper():
        all_bulb_counts.update(counts)

# --------------------------
# Aggregate plot function
# --------------------------
def plot_aggregate(counter, title, known_classes=['stop', 'go', 'warning']):
    # Group other/outlier classes
    grouped = {}
    other_count = 0
    for cls, c in counter.items():
        if cls in known_classes:
            grouped[cls] = c
        else:
            other_count += c
    grouped['other'] = other_count

    # Plot
    plt.figure(figsize=(6,4))
    plt.bar(grouped.keys(), grouped.values(), color='skyblue')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Total Count")
    plt.show()

# --------------------------
# Plot aggregate BOX and BULB distributions
# --------------------------
plot_aggregate(all_box_counts, "Total BOX Annotations Across Dataset")
plot_aggregate(all_bulb_counts, "Total BULB Annotations Across Dataset")

print("\n=== Parts 1 & 2 Completed Successfully ===")
