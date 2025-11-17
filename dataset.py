import os
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import zipfile

# --------------------------
# Authenticate Kaggle API
# --------------------------
api = KaggleApi()
api.authenticate()

# --------------------------
# Download dataset
# --------------------------
dataset_name = "mbornoe/lisa-traffic-light-dataset"  
destination_folder = "./lisa_dataset"  
os.makedirs(destination_folder, exist_ok=True)

print(f"Downloading dataset: {dataset_name}")

# Download without unzipping first
api.dataset_download_files(dataset_name, path=destination_folder, unzip=False, quiet=False)

# Locate the downloaded zip
zip_path = os.path.join(destination_folder, f"{dataset_name.split('/')[-1]}.zip")
print(f"Downloaded zip: {zip_path}")
print("Extracting files with progress bar...")

# Extract with progress bar
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
        zip_ref.extract(file, destination_folder)

print(f"Dataset downloaded and extracted to: {destination_folder}")
