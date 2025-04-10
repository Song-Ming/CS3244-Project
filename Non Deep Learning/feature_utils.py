import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# Shape
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _= hog(gray, pixels_per_cell=(32, 32), cells_per_block=(2, 2),
                  orientations=6, block_norm='L2-Hys', visualize=True)
    return features

# Colour
def extract_color_hist(image, bins=32):
    hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()
    hist = np.concatenate([hist_b, hist_g, hist_r])
    return hist / np.sum(hist)

# Texture
def extract_lbp(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_all_features(image):
    return np.concatenate([
        extract_hog(image),
        extract_color_hist(image),
        extract_lbp(image)
    ])
