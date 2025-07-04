import cv2
from sklearn.cluster import KMeans

def generate_cloud_mask(img, k=2, resize_factor=0.25):
    original_shape = img.shape
    img_small = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    pixels = img_small.reshape((-1, 1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    labels = kmeans.labels_.reshape(img_small.shape)
    cloud_cluster = np.argmax([np.mean(img_small[labels == i]) for i in range(k)])
    mask_small = (labels == cloud_cluster).astype(np.uint8) * 255
    return cv2.resize(mask_small, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
