import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew

# =====================================================
# CONFIG
# =====================================================

DATASET = "ISIC2019"   # "HAM10000" or "ISIC2019"
CLEAN_DATA_ROOT = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Code\CleanData"

# =====================================================
# DATASET PATHS & CLASS MAPS
# =====================================================

if DATASET == "HAM10000":
    IMAGE_DIR = os.path.join(CLEAN_DATA_ROOT, "HAM10000", "images")
    LABEL_CSV = os.path.join(CLEAN_DATA_ROOT, "HAM10000", "HAM10000_metadata.csv")

    CLASS_MAP = {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6
    }

    RAW_OUT = "ham10000_color_raw.csv"
    NORM_OUT = "ham10000_color_norm.csv"

elif DATASET == "ISIC2019":
    IMAGE_DIR = os.path.join(CLEAN_DATA_ROOT, "ISIC2019", "images_train")
    LABEL_CSV = os.path.join(
        CLEAN_DATA_ROOT,
        "ISIC2019",
        "ISIC_2019_Training_GroundTruth.csv"
    )

    CLASS_MAP = {
        "AK": 0,
        "BCC": 1,
        "BKL": 2,
        "DF": 3,
        "MEL": 4,
        "NV": 5,
        "SCC": 6,
        "VASC": 7
    }

    RAW_OUT = "isic2019_color_raw.csv"
    NORM_OUT = "isic2019_color_norm.csv"

else:
    raise ValueError("DATASET must be 'HAM10000' or 'ISIC2019'")

print(f"\nRunning color feature extraction for: {DATASET}")
print("Image dir :", IMAGE_DIR)
print("Label CSV :", LABEL_CSV)

# =====================================================
# COLOR FEATURE FUNCTIONS
# =====================================================

def color_stats(img):
    feats = []

    # RGB stats
    for c in cv2.split(img):
        feats.append(np.mean(c))
        feats.append(np.std(c))

    # HSV stats
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for c in cv2.split(hsv):
        feats.append(np.mean(c))
        feats.append(np.std(c))

    return feats


def color_moments(img):
    feats = []
    for c in cv2.split(img):
        feats.append(np.mean(c))
        feats.append(np.std(c))
        feats.append(skew(c.reshape(-1)))
    return feats


def extract_color_features(img):
    feats = []
    feats.extend(color_stats(img))
    feats.extend(color_moments(img))
    return np.array(feats, dtype=np.float32)

# =====================================================
# LOAD LABELS
# =====================================================

labels_df = pd.read_csv(LABEL_CSV)
print("Label columns:", labels_df.columns.tolist())

# =====================================================
# FEATURE EXTRACTION
# =====================================================

rows = []

for _, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                   desc="Extracting color features"):

    if DATASET == "HAM10000":
        image_id = row["image_id"]
        label = CLASS_MAP[row["dx"]]

    else:  # ISIC2019
        image_id = row["image"]
        label_name = max(CLASS_MAP, key=lambda c: row[c])
        label = CLASS_MAP[label_name]

    img_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    color_feat = extract_color_features(img)
    rows.append(np.concatenate([color_feat, [label]]))

# =====================================================
# SAVE RAW FEATURES
# =====================================================

feature_names = [f"color_{i}" for i in range(len(color_feat))] + ["label"]
raw_df = pd.DataFrame(rows, columns=feature_names)
raw_df.to_csv(RAW_OUT, index=False)

print(f"\nRAW color features saved: {RAW_OUT}")
print(f"Shape: {raw_df.shape}")

# =====================================================
# FEATURE-WISE MIN–MAX NORMALIZATION
# =====================================================

norm_df = raw_df.copy()
feat_cols = [c for c in norm_df.columns if c != "label"]

for col in feat_cols:
    min_val = norm_df[col].min()
    max_val = norm_df[col].max()
    if max_val > min_val:
        norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
    else:
        norm_df[col] = 0.0

norm_df.to_csv(NORM_OUT, index=False)

print(f"\nNORMALIZED color features saved: {NORM_OUT}")
print(f"Shape: {norm_df.shape}")
