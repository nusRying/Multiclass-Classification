import os
import sys
import time
import json
import datetime
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# =====================================================
# PATH SETUP
# =====================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXSTRACS_ROOT = os.path.join(PROJECT_ROOT, "scikit-ExSTraCS-master")
if EXSTRACS_ROOT not in sys.path:
    sys.path.insert(0, EXSTRACS_ROOT)

from skExSTraCS.ExSTraCS import ExSTraCS

DATA_ROOT = os.path.dirname(__file__)
# =====================================================
# METRICS (MULTICLASS SAFE)
# =====================================================

def compute_metrics(y_true, y_pred):
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return bal_acc, cm

# =====================================================
# CROSS-VALIDATION FUNCTION
# =====================================================

def run_cv(
    csv_path,
    dataset_name,
    feature_set_name,
    param_grid,
    n_splits=5,
    out_dir="lcs"
):
    print(f"\n=== DATASET: {dataset_name} | FEATURES: {feature_set_name} ===")

    data = pd.read_csv(os.path.join(DATA_ROOT, csv_path))

    # Label is LAST column
    X = data.iloc[:, :-1].values.astype(float)
    y = data.iloc[:, -1].values.astype(int)

    print(
        f"Feature range: min={np.min(X):.4f}, max={np.max(X):.4f} "
        "(pre-normalized)"
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    per_fold_records = []
    summary_records = []

    for params in param_grid:
        print("Params:", params)
        fold_scores = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
            seed = 42 + fold

            # -------------------------
            # NO NORMALIZATION HERE
            # -------------------------
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            # -------------------------
            # MODEL SETUP
            # -------------------------
            model = ExSTraCS()

            model.N = params.get("N", 2000)
            model.learningIterations = params.get("learningIterations", 100000)
            model.theta_sel = params.get("theta_sel", 0.8)

            # Minority & specificity pressure
            model.nu = params.get("nu", 3.0)
            model.p_spec = params.get("p_spec", 0.4)

            # GA parameters
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
                "feature_set": feature_set_name,
                "params": params,
                "fold": fold,
                "balanced_accuracy": bal_acc,
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "rule_population": pop_size,
                "duration_seconds": round(duration, 3),
                "fit_exception": fit_exception
            })

            if bal_acc is not None:
                fold_scores.append(bal_acc)

        summary_records.append({
            "dataset": dataset_name,
            "feature_set": feature_set_name,
            "params": params,
            "mean_bal_acc": float(np.mean(fold_scores)) if fold_scores else None,
            "std_bal_acc": float(np.std(fold_scores)) if fold_scores else None,
            "timestamp": datetime.datetime.now().isoformat()
        })

    # =================================================
    # SAVE RESULTS
    # =================================================

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    folds_path = os.path.join(
        out_dir,
        f"exstracs_{dataset_name}_{feature_set_name}_folds_{ts}.jsonl"
    )
    summary_path = os.path.join(
        out_dir,
        f"exstracs_{dataset_name}_{feature_set_name}_summary_{ts}.json"
    )

    with open(folds_path, "w", encoding="utf-8") as fh:
        for rec in per_fold_records:
            fh.write(json.dumps(rec) + "\n")

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_records, fh, indent=2)

    print(f"Saved folds   -> {folds_path}")
    print(f"Saved summary -> {summary_path}")

    return pd.DataFrame(summary_records)

# =====================================================
# MAIN: COLOR FUSION EXPERIMENTS
# =====================================================

if __name__ == "__main__":

    param_grid = [
        {"N": 1500, "learningIterations": 100000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.8},
        {"N": 2000, "learningIterations": 120000, "theta_sel": 0.9},
    ]

    experiments = [
        ("ham10000_color_norm.csv", "HAM10000", "Color"),
        ("isic2019_color_norm.csv", "ISIC2019", "Color"),
    ]

    out_dir = os.path.join(PROJECT_ROOT, "lcs", "color_no_foldnorm")
    all_runs = []

    for csv_path, dataset, feature_set in experiments:
        df = run_cv(
            csv_path=csv_path,
            dataset_name=dataset,
            feature_set_name=feature_set,
            param_grid=param_grid,
            n_splits=5,
            out_dir=out_dir
        )
        all_runs.append(df)

    final_df = pd.concat(all_runs, ignore_index=True)

    final_out = os.path.join(
        out_dir,
        f"exstracs_color_fusion_comparison_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    final_df.to_csv(final_out, index=False)
    print(f"\nALL DONE. Final comparison CSV saved to:\n{final_out}")
