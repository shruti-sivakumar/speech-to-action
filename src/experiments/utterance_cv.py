"""
Speaker-grouped 5-fold cross-validation for the utterance-level classifiers
(prosodic statistics and prosodic + MFCC statistics, SVM and MLP).

Puts the utterance-level results on the same evaluation protocol as the
sequence models, and adds the reduced-class slot tasks: the full 31-class
intent plus action (6), object (14) and location (4) as separate targets.

Fold assignment is taken from GroupKFold on speakerId over the same 30,043
utterances used by the sequence-model scripts, so every speaker sits in the
same held-out fold as in cnn_logmel_cv.py / wav2vec2_sequence_cv.py.
Hyperparameters are the configurations selected on the official validation
split (Notebook 06/07 grid search) and are held fixed across folds and targets.
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from src.config import DATA_ROOT, RESULTS_ROOT

N_SPLITS = 5
N_MFCC = 13
TARGETS = ["intent", "action", "object", "location"]

PROSODIC = ["mean_energy_db", "std_energy_db", "mean_zcr", "std_zcr", "voiced_fraction",
            "mean_f0", "std_f0", "mean_f1", "mean_f2", "duration_s"]
WITH_MFCC = PROSODIC + [f"mfcc{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "std")]

# (feature set name, columns, SVM params, MLP params) -- selected on the official validation split
CONFIGS = [
    ("prosodic", PROSODIC, dict(C=10, gamma=0.01), dict(hidden_layer_sizes=(64,), alpha=1e-3)),
    ("prosodic_mfcc", WITH_MFCC, dict(C=1, gamma=0.01), dict(hidden_layer_sizes=(64, 32), alpha=1e-3)),
]


def run_fold(fold, train_idx, held_idx, X, y, svm_params, mlp_params):
    scaler = StandardScaler().fit(X[train_idx])
    X_tr, X_ho = scaler.transform(X[train_idx]), scaler.transform(X[held_idx])
    majority = np.bincount(y[train_idx]).argmax()
    svm = SVC(kernel="rbf", **svm_params).fit(X_tr, y[train_idx])
    mlp = MLPClassifier(max_iter=1000, random_state=0, **mlp_params).fit(X_tr, y[train_idx])
    return {
        "fold": fold + 1,
        "majority_acc": (y[held_idx] == majority).mean(),
        "svm_acc": (svm.predict(X_ho) == y[held_idx]).mean(),
        "mlp_acc": (mlp.predict(X_ho) == y[held_idx]).mean(),
        "n_train": len(train_idx), "n_held_out": len(held_idx),
    }


def main():
    feats = pd.read_csv(f"{DATA_ROOT}/features_cache_mfcc.csv")
    feats["intent"] = feats["action"] + "_" + feats["object"] + "_" + feats["location"]
    speaker = feats["speakerId"].values

    # Same speaker-to-fold assignment as the sequence-model scripts
    folds = list(GroupKFold(n_splits=N_SPLITS).split(feats, groups=speaker))

    rows = []
    t_start = time.time()
    for name, cols, svm_params, mlp_params in CONFIGS:
        X = feats[cols].values
        for target in TARGETS:
            y = LabelEncoder().fit_transform(feats[target].values)
            print(f"=== {name} / {target} ({len(np.unique(y))} classes) ===", flush=True)
            res = Parallel(n_jobs=N_SPLITS)(
                delayed(run_fold)(f, tr, ho, X, y, svm_params, mlp_params)
                for f, (tr, ho) in enumerate(folds))
            for r in res:
                r.update(features=name, target=target, n_classes=len(np.unique(y)))
                print(f"  fold {r['fold']}: majority {r['majority_acc']:.4f}  "
                      f"SVM {r['svm_acc']:.4f}  MLP {r['mlp_acc']:.4f}", flush=True)
            rows.extend(res)
            pd.DataFrame(rows).to_csv(f"{RESULTS_ROOT}/cv_utterance_all_targets.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["features", "target"])[["majority_acc", "svm_acc", "mlp_acc"]].agg(["mean", "std"])
    print("\n=== SUMMARY (mean / std over folds) ===")
    print(summary.round(4).to_string())
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
