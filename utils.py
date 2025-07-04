import pandas as pd
import numpy as np

def add_engineered_features(df):
    df['area_bbox_ratio'] = df['area'] / df['bbox_area']
    df['centroid_magniterude'] = np.sqrt(df['centroid_x']**2 + df['centroid_y']**2)
    df['area_std_cluster'] = df.groupby('clust_id')['area'].transform('std').fillna(0)
    df['motion_std_cluster'] = df.groupby('cluster_id')['motion_magnitude'].transform('std').fillna(0)
    df['shape_index'] = df['solidity'] * df['extent']
    return df