import os
import sys
import time
import json
import datetime
import traceback

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# =====================================================
# PATH SETUP
# =====================================================

EXSTRACS_ROOT = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Code\scikit-ExSTraCS-master"
DATA_ROOT = r"C:\Users\umair\Videos\PhD\PhD Data\Week 8 Jannuary\Multiclass Classification"

if EXSTRACS_ROOT not in sys.path:
    sys.path.insert(0, EXSTRACS_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS

# =====================================================
# METRICS
# =====================================================

def compute_metrics(y_true, y_pred):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return bal_acc, cm

# =====================================================
# MIN–MAX NORMALIZATION (TRAIN-ONLY)
# =====================================================

def minmax_fit(X):
    """
    Fit Min–Max scaler on training data only.
    """
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return min_val, max_val


def minmax_transform(X, min_val, max_val):
    """
    Apply Min–Max normalization using precomputed stats.
    """
    denom = (max_val - min_val)
    denom[denom == 0] = 1.0  # prevent division by zero
    return (X - min_val) / denom

# =====================================================
# CROSS-VALIDATION CORE
# =====================================================

def run_cv(csv_path, dataset_name, feature_family, param_grid,
           n_splits=5, out_dir="lcs"):

    print(f"\n=== {dataset_name} | {feature_family} (Fold-wise Min–Max Norm) ===")

    csv_path = os.path.join(DATA_ROOT, csv_path)
    data = pd.read_csv(csv_path)

    feature_cols = [c for c in data.columns if c not in ("image", "label")]
    X = data[feature_cols].values.astype(float)
    y = data["label"].values.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []
    per_fold_records = []

    for params in param_grid:
        print("Params:", params)
        fold_scores = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
            seed = 42 + fold

            # =========================
            # SPLIT
            # =========================
            X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # =========================
            # FOLD-WISE NORMALIZATION
            # =========================
            min_val, max_val = minmax_fit(X_tr_raw)
            X_tr = minmax_transform(X_tr_raw, min_val, max_val)
            X_te = minmax_transform(X_te_raw, min_val, max_val)

            # =========================
            # MODEL SETUP
            # =========================
            model = ExSTraCS()
            model.N = params.get("N", 2000)
            model.learningIterations = params.get("learningIterations", 100000)
            model.theta_sel = params.get("theta_sel", 0.8)

            # Minority & specificity bias
            model.nu = params.get("nu", 3.0)
            model.p_spec = params.get("p_spec", 0.4)
            model.theta_GA = params.get("theta_GA", 15)
            model.chi = params.get("chi", 0.8)
            model.mu = params.get("mu", 0.04)

            model.doSubsumption = True
            model.useBalancedAccuracy = True
            model.randomSeed = seed

            print(
                f"  Fold {fold} | seed={seed} "
                f"N={model.N} iters={model.learningIterations}"
            )

            start = time.time()
            fit_exception = None

            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                bal_acc, cm = compute_metrics(y_te, y_pred)
            except Exception:
                fit_exception = traceback.format_exc()
                bal_acc = None
                cm = None

            duration = time.time() - start

            try:
                pop_size = len(model.population.popSet)
            except Exception:
                pop_size = None

            print(
                f"    BA={bal_acc} | time={duration:.1f}s | rules={pop_size}"
            )

            per_fold_records.append({
                "dataset": dataset_name,
                "feature_family": feature_family,
                "params": params,
                "fold": fold,
                "balanced_accuracy": bal_acc,
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "duration_seconds": round(duration, 3),
                "rule_population": pop_size,
                "fit_exception": fit_exception
            })

            if bal_acc is not None:
                fold_scores.append(bal_acc)

        all_results.append({
            "dataset": dataset_name,
            "feature_family": feature_family,
            "params": params,
            "mean_bal_acc": float(np.mean(fold_scores)) if fold_scores else None,
            "std_bal_acc": float(np.std(fold_scores)) if fold_scores else None,
            "timestamp": datetime.datetime.now().isoformat()
        })

    # =========================
    # SAVE RESULTS
    # =========================
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    folds_path = os.path.join(
        out_dir,
        f"exstracs_{dataset_name}_{feature_family}_foldNorm_folds_{ts}.jsonl"
    )
    summary_path = os.path.join(
        out_dir,
        f"exstracs_{dataset_name}_{feature_family}_foldNorm_summary_{ts}.json"
    )

    with open(folds_path, "w") as fh:
        for rec in per_fold_records:
            fh.write(json.dumps(rec) + "\n")

    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    print(f"Saved folds   -> {folds_path}")
    print(f"Saved summary -> {summary_path}")

    return pd.DataFrame(all_results)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    param_grid = [
        {"N": 1500, "learningIterations": 100000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.9},
    ]

    experiments = [
        ("ham10000_lbp_multiclass_clean_norm.csv",  "HAM10000", "LBP"),
        ("ham10000_glcm_multiclass_clean_norm.csv", "HAM10000", "GLCM"),
        ("isic2019_lbp_multiclass_clean_norm.csv",  "ISIC2019", "LBP"),
        ("isic2019_glcm_multiclass_clean_norm.csv", "ISIC2019", "GLCM"),
    ]

    out_dir = os.path.join(DATA_ROOT, "lcs")

    all_runs = []

    for csv_path, dataset, feature_family in experiments:
        df = run_cv(
            csv_path=csv_path,
            dataset_name=dataset,
            feature_family=feature_family,
            param_grid=param_grid,
            n_splits=5,
            out_dir=out_dir
        )
        all_runs.append(df)

    final_df = pd.concat(all_runs, ignore_index=True)

    final_out = os.path.join(
        out_dir,
        f"exstracs_feature_family_comparison_foldNorm_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    final_df.to_csv(final_out, index=False)
    print(f"\nALL DONE. Final comparison CSV saved to:\n{final_out}")
