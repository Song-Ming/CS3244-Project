import os
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from paths import IMAGES_DIR, LABEL_CSV, OUTPUT_PATH
from feature_utils import extract_all_features

def main():
    if not LABEL_CSV.exists():
        print(f"Missing labels.csv at: {LABEL_CSV}")
        return

    df = pd.read_csv(LABEL_CSV)
    X, y = [], []

    for i, row in df.iterrows():
        filename = row['Filename']
        label = row['Label']
        img_path = IMAGES_DIR / filename

        if not img_path.exists():
            print(f"[{i}] Skipping missing image: {filename}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i}] Failed to load image: {filename}")
            continue

        try:
            features = extract_all_features(image)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"[{i}] Error processing {filename}: {e}")

        if i % 1000 == 0: # Just checking every 1000 to see if the data is processing properly
            print(f"Processed {i}/{len(df)} images...")

    X = np.array(X)
    y = np.array(y)

    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Process completed. Saved features to: {OUTPUT_PATH}")
    print(f"X shape: {X.shape}, Y shape: {y.shape}")

if __name__ == "__main__":
    main()
