# feature_extractor.py
import cv2
import numpy as np
from PIL import Image

def extract_features(image_path):
    """
    Extract color, texture, and shape features from a leaf image.
    Returns a flat feature vector.
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))

    features = []

    # ── 1. COLOR FEATURES (HSV histogram) ──────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    # → 96 features

    # ── 2. TEXTURE FEATURES (GLCM via grayscale stats) ─────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP-like: mean, std of gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    features.extend([
        np.mean(magnitude), np.std(magnitude),
        np.mean(gray), np.std(gray),
        np.percentile(gray, 25), np.percentile(gray, 75)
    ])
    # → 6 features

    # ── 3. SHAPE FEATURES (Hu Moments) ─────────────────────────
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    features.extend(hu)
    # → 7 features

    # ── 4. GREEN CHANNEL STATS (plant health indicator) ─────────
    green = img[:, :, 1]
    features.extend([
        np.mean(green), np.std(green),
        np.sum(green > 100) / green.size   # green pixel ratio
    ])
    # → 3 features

    return np.array(features, dtype=np.float32)
    # TOTAL: 112 features


def extract_features_from_array(img_array):
    """
    Same as above but accepts a numpy array (for Streamlit uploaded files).
    """
    img = cv2.resize(img_array, (128, 128))
    # Convert RGB (PIL) → BGR (OpenCV)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    features = []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    features.extend([
        np.mean(magnitude), np.std(magnitude),
        np.mean(gray), np.std(gray),
        np.percentile(gray, 25), np.percentile(gray, 75)
    ])

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    features.extend(hu)

    green = img[:, :, 1]
    features.extend([
        np.mean(green), np.std(green),
        np.sum(green > 100) / green.size
    ])

    return np.array(features, dtype=np.float32)