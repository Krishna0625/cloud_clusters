from scipy.spatial.distance import cdist
def track_cloud_clusters(df):
    df['cluster_id'] = None
    next_cluster_id = 0
    prev_frame = None

    for ts in sorted(df['timestamp'].unique()):
        curr_frame = df[df['timestamp'] == ts].copy()

        if prev_frame is None:
            curr_frame['cluster_id'] = list(range(next_cluster_id, next_cluster_id + len(curr_frame)))
            next_cluster_id += len(curr_frame)
        else:
            dist_matrix = cdist(prev_frame[['centroid_x', 'centroid_y']], curr_frame[['centroid_x', 'centroid_y']])
            curr_frame['cluster_id'] = None
            for curr_idx, row in enumerate(curr_frame.itertuples(index=False)):
                min_idx = dist_matrix[:, curr_idx].argmin()
                if dist_matrix[min_idx, curr_idx] < 50:
                    curr_frame.iloc[curr_idx, curr_frame.columns.get_loc('cluster_id')] = prev_frame.iloc[min_idx]['cluster_id']
                else:
                    curr_frame.iloc[curr_idx, curr_frame.columns.get_loc('cluster_id')] = next_cluster_id
                    next_cluster_id += 1

        df.loc[curr_frame.index, 'cluster_id'] = curr_frame['cluster_id']
        prev_frame = curr_frame
    return df