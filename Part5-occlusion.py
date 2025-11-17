# EDA_LISA_BBox_Full.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------
# 0. Dataset path
# --------------------------
dataset_path = "./lisa_dataset"  # Adjust if needed
annotations_folder = os.path.join(dataset_path, "Annotations")

# --------------------------
# 1. Find all annotation CSVs recursively
# --------------------------
annotation_files = glob.glob(os.path.join(annotations_folder, "**", "*.csv"), recursive=True)
if not annotation_files:
    raise FileNotFoundError(f"No CSV annotation files found in {annotations_folder} or its subfolders!")

print(f"Found {len(annotation_files)} CSV files.")
for f in annotation_files:
    print(" -", f)

# --------------------------
# 2. Load all boxes
# --------------------------
all_boxes = []

for ann_file in annotation_files:
    df_name = os.path.basename(ann_file).replace(".csv", "")
    
    df = pd.read_csv(ann_file, header=None)
    df_fixed = df[0].astype(str).str.strip().str.replace('"','').str.split(";", expand=True)
    
    if df_fixed.shape[1] < 10:
        continue
    df_fixed = df_fixed.iloc[:, :10]
    df_fixed.columns = [
        "Filename","Annotation_tag","Upper_left_X","Upper_left_Y",
        "Lower_right_X","Lower_right_Y","Origin_file",
        "Origin_frame_number","Origin_track","Origin_track_frame_number"
    ]
    
    # Convert coordinates to numeric, drop invalid rows
    for col in ["Upper_left_X","Upper_left_Y","Lower_right_X","Lower_right_Y"]:
        df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
    df_fixed = df_fixed.dropna(subset=["Upper_left_X","Upper_left_Y","Lower_right_X","Lower_right_Y"])
    
    for idx, row in df_fixed.iterrows():
        width = row["Lower_right_X"] - row["Upper_left_X"]
        height = row["Lower_right_Y"] - row["Upper_left_Y"]
        area = width * height
        aspect_ratio = width / (height + 1e-5)
        all_boxes.append({
            "Filename": row["Filename"],
            "Class": row["Annotation_tag"],
            "Width": width,
            "Height": height,
            "Area": area,
            "Aspect_Ratio": aspect_ratio
        })

# --------------------------
# 3. Create DataFrame
# --------------------------
boxes_df = pd.DataFrame(all_boxes)
print(f"\nTotal bounding boxes analyzed: {len(boxes_df)}")

if boxes_df.empty:
    raise ValueError("No valid bounding boxes found. Check CSV formatting and paths!")

# --------------------------
# 4. Summary statistics
# --------------------------
print("\nBounding Box Summary Statistics:")
print(boxes_df[["Width", "Height", "Area", "Aspect_Ratio"]].describe())

# --------------------------
# 5. Identify small / large / unusual boxes
# --------------------------
small_threshold = 20
large_threshold = 500
boxes_df['Small_Box'] = (boxes_df['Width'] < small_threshold) | (boxes_df['Height'] < small_threshold)
boxes_df['Large_Box'] = (boxes_df['Width'] > large_threshold) | (boxes_df['Height'] > large_threshold)
boxes_df['Unusual_Aspect_Ratio'] = (boxes_df['Aspect_Ratio'] < 0.5) | (boxes_df['Aspect_Ratio'] > 2.0)

print(f"\nSmall boxes (<{small_threshold}px): {boxes_df['Small_Box'].sum()}")
print(f"Large boxes (>{large_threshold}px): {boxes_df['Large_Box'].sum()}")
print(f"Boxes with unusual aspect ratio: {boxes_df['Unusual_Aspect_Ratio'].sum()}")

# --------------------------
# 6. Plots
# --------------------------
plt.figure(figsize=(8,5))
plt.hist(boxes_df['Area'], bins=50, color='skyblue', edgecolor='black')
plt.title("Bounding Box Area Distribution")
plt.xlabel("Area (pixels)")
plt.ylabel("Number of Boxes")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(boxes_df['Aspect_Ratio'], bins=50, color='lightgreen', edgecolor='black')
plt.title("Bounding Box Aspect Ratio Distribution")
plt.xlabel("Width / Height")
plt.ylabel("Number of Boxes")
plt.show()

# Class-level box count
class_counts = Counter(boxes_df['Class'])
plt.figure(figsize=(6,4))
plt.bar(class_counts.keys(), class_counts.values(), color='orange')
plt.title("Bounding Box Count per Class")
plt.xlabel("Class")
plt.ylabel("Number of Boxes")
plt.show()

# Save boxes_df to disk
output_path = os.path.join(dataset_path, "boxes_df.pkl")
boxes_df.to_pickle(output_path)
print(f"boxes_df saved to {output_path}")
