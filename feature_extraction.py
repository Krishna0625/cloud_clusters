import pandas as pd
from skimage.measure import label, regionprops

def extract_cloud_features(mask, filename):
    labeled_mask = label(mask > 0)
    props = regionprops(labeled_mask)
    features = []
    for prop in props:
        if prop.area < 500:
            continue
        features.append({
            'file': filename,
            'area': prop.area,
            'centroid_x': round(prop.centroid[1], 2),
            'centroid_y': round(prop.centroid[0], 2),
            'bbox_area': prop.bbox_area,
            'extent': round(prop.extent, 3),
            'solidity': round(prop.solidity, 3)
        })
    return features

