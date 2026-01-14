import pandas as pd
import os

data_root = r"c:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Multiclass Classification"
files = ["ham10000_color_raw.csv", "isic2019_color_raw.csv"]

print("--- DATASET COMPOSITION ---")
for f_name in files:
    path = os.path.join(data_root, f_name)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Assuming label is the last column
            label_col = df.columns[-1]
            counts = df[label_col].value_counts().sort_index()
            with open("counts_utf8.txt", "a", encoding="utf-8") as f:
                f.write(f"\nDataset: {f_name}\n")
                f.write(f"Total samples: {len(df)}\n")
                f.write(f"Class distribution:\n{counts}\n")
                f.write("-" * 30 + "\n")
            print(f"Processed {f_name}")
        except Exception as e:
            print(f"Error reading {f_name}: {e}")
    else:
        print(f"File not found: {path}")
