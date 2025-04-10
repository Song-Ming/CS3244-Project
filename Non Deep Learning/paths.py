# CS3244-Project/
# └── data/
#     ├── images/
#     └── models/
#          └── features.pkl/ (feature extraction)
#     └── labels.csv
# Follow this path order, data/ should be in .gitignore

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LABEL_CSV = DATA_DIR / "labels.csv"
OUTPUT_PATH = DATA_DIR / "models" / "features.pkl"
