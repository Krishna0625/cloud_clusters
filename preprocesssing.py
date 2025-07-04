import os
import rasterio
import cv2
import numpy as np

def convert_tif_to_png(tif_path, band_index=1):
    with rasterio.open(tif_path) as src:
        data = src.read(band_index)
        norm = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        return norm.astype(np.uint8)

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

