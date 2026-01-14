import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pywt

# =====================================================
# CONFIG
# =====================================================

DATASET = "ISIC2019"   # "HAM10000" or "ISIC2019"
CLEAN_DATA_ROOT = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Code\CleanData"

WAVELET = "db2"
LEVEL = 2

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

    RAW_OUT = "ham10000_wavelet_raw.csv"
    NORM_OUT = "ham10000_wavelet_norm.csv"

elif DATASET == "ISIC2019":
    IMAGE_DIR = os.path.join(CLEAN_DATA_ROOT, "ISIC2019", "images_train")
    LABEL_CSV = os.path.join(CLEAN_DATA_ROOT, "ISIC2019", "ISIC_2019_Training_GroundTruth.csv")

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

    RAW_OUT = "isic2019_wavelet_raw.csv"
    NORM_OUT = "isic2019_wavelet_norm.csv"

else:
    raise ValueError("DATASET must be 'HAM10000' or 'ISIC2019'")

# =====================================================
# PATH RESOLUTION
# =====================================================

def resolve_label_csv(path):
    if os.path.exists(path):
        return path
    no_ext = os.path.splitext(path)[0]
    if os.path.exists(no_ext):
        return no_ext
    raise FileNotFoundError(f"Label CSV not found: {path} or {no_ext}")

LABEL_CSV = resolve_label_csv(LABEL_CSV)

# =====================================================
# WAVELET FEATURE FUNCTION
# =====================================================

def extract_wavelet_features(gray):
    """
    Extract wavelet energy features from sub-bands
    """
    coeffs = pywt.wavedec2(gray, wavelet=WAVELET, level=LEVEL)

    features = []

    # Approximation coefficients (low frequency)
    cA = coeffs[0]
    features.append(np.mean(np.abs(cA)))
    features.append(np.std(cA))

    # Detail coefficients (high frequency)
    for level_coeffs in coeffs[1:]:
        cH, cV, cD = level_coeffs
        for band in (cH, cV, cD):
            features.append(np.mean(np.abs(band)))
            features.append(np.std(band))

    return np.array(features, dtype=np.float32)

# =====================================================
# LOAD LABELS
# =====================================================

labels_df = pd.read_csv(LABEL_CSV)

# =====================================================
# FEATURE EXTRACTION
# =====================================================

rows = []

for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting wavelet features"):

    if DATASET == "HAM10000":
        image_id = row["image_id"]
        label = CLASS_MAP[row["dx"]]
    else:
        image_id = row["image"]
        label_name = max(CLASS_MAP, key=lambda c: row[c])
        label = CLASS_MAP[label_name]

    img_path = os.path.join(IMAGE_DIR, image_id + ".jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wavelet_feat = extract_wavelet_features(gray)
    rows.append(np.concatenate([wavelet_feat, [label]]))

# =====================================================
# SAVE RAW WAVELET FEATURES
# =====================================================

feature_names = [f"wav_{i}" for i in range(len(wavelet_feat))] + ["label"]
raw_df = pd.DataFrame(rows, columns=feature_names)
raw_df.to_csv(RAW_OUT, index=False)

print(f"\nRAW wavelet features saved: {RAW_OUT}")
print(f"Shape: {raw_df.shape}")

# =====================================================
# FEATURE-WISE MIN-MAX NORMALIZATION
# =====================================================

norm_df = raw_df.copy()
feature_cols = [c for c in norm_df.columns if c != "label"]

for col in feature_cols:
    min_val = norm_df[col].min()
    max_val = norm_df[col].max()
    if max_val > min_val:
        norm_df[col] = (norm_df[col] - min_val) / (max_val - min_val)
    else:
        norm_df[col] = 0.0

norm_df.to_csv(NORM_OUT, index=False)

print(f"\nNORMALIZED wavelet features saved: {NORM_OUT}")
print(f"Shape: {norm_df.shape}")
