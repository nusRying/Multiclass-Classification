import os
import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Multiclass Classification\isic2019_color_raw.csv"
OUTPUT_CSV = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Multiclass Classification\isic2019_color_norm.csv"

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_CSV)

# Label is LAST column (as you stated)
label_col = df.columns[-1]
feature_cols = df.columns[:-1]

print("Input file :", INPUT_CSV)
print("Features   :", len(feature_cols))
print("Label col  :", label_col)

# =====================================================
# FEATURE-WISE MIN–MAX NORMALIZATION
# =====================================================

norm_df = df.copy()

for col in feature_cols:
    min_val = df[col].min()
    max_val = df[col].max()

    if max_val > min_val:
        norm_df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        # Constant feature → zeroed
        norm_df[col] = 0.0

# =====================================================
# SANITY CHECK
# =====================================================

print(
    f"Normalized feature range: "
    f"min={norm_df[feature_cols].min().min():.4f}, "
    f"max={norm_df[feature_cols].max().max():.4f}"
)

# =====================================================
# SAVE
# =====================================================

norm_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Normalized file saved to:\n{OUTPUT_CSV}")
print(f"Shape: {norm_df.shape}")
