# main.py

import os
import cv2
import pandas as pd
from datetime import datetime
from src.preprocessing import convert_tif_to_png, save_image
from src.mask_generation import generate_cloud_mask
from src.feature_extraction import extract_cloud_features
from src.tracking import track_cloud_clusters
from src.utils import add_engineered_features
from src.model_train import train_model

# === CONFIGURATION ===
DATA_DIR = "data"
MASK_DIR = "masks"
FEATURES_CSV = "features/cloud_features.csv"
MODEL_PATH = "models/cloud_cluster_model.pkl"
TIMESTAMP_FORMAT = "%H%M"  # e.g. 0015, 0115

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs("features", exist_ok=True)
os.makedirs("models", exist_ok=True)

# === STEP 1: Convert TIFs to PNGs ===
tif_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
for tif_file in tif_files:
    tif_path = os.path.join(DATA_DIR, tif_file)
    png_image = convert_tif_to_png(tif_path)
    output_png_path = os.path.join(DATA_DIR, tif_file.replace(".tif", ".png"))
    save_image(png_image, output_png_path)

# === STEP 2: Generate Cloud Masks ===
image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
for file in image_files:
    img = cv2.imread(os.path.join(DATA_DIR, file), cv2.IMREAD_GRAYSCALE)
    mask = generate_cloud_mask(img)
    save_image(mask, os.path.join(MASK_DIR, file.replace(".png", "_cloudmask.png")))

# === STEP 3: Extract Cloud Features ===
feature_rows = []
for file in os.listdir(MASK_DIR):
    if not file.endswith("_cloudmask.png"):
        continue
    mask = cv2.imread(os.path.join(MASK_DIR, file), cv2.IMREAD_GRAYSCALE)
    rows = extract_cloud_features(mask, file)
    for row in rows:
        # Extract timestamp from filename
        timestamp = row['file'].split('_')[0]
        row['timestamp'] = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        feature_rows.append(row)

df = pd.DataFrame(feature_rows)

# === STEP 4: Track Cloud Clusters ===
df = track_cloud_clusters(df)

# === STEP 5: Engineer Features ===
df['area_diff'] = df.groupby('cluster_id')['area'].diff().fillna(0)
df['dx'] = df.groupby('cluster_id')['centroid_x'].diff().fillna(0)
df['dy'] = df.groupby('cluster_id')['centroid_y'].diff().fillna(0)
df['motion_magnitude'] = (df['dx']**2 + df['dy']**2)**0.5
df = add_engineered_features(df)

# === STEP 6: Label by Area (for Supervised Learning) ===
threshold = df['area'].median()
df['label'] = (df['area'] > threshold).astype(int)

# === STEP 7: Train Model ===
X = df[['area', 'extent', 'solidity', 'bbox_area', 'area_diff', 'motion_magnitude']]
y = df['label']
train_model(X, y, MODEL_PATH)

# === Save Final Features ===
df.to_csv(FEATURES_CSV, index=False)
print(f"\n✅ Pipeline complete. Features saved to {FEATURES_CSV}")
