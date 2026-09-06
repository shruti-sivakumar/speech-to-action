"""
Stage 9 speaker-grouped k-fold cross-validation: frozen wav2vec2-base
embeddings + a small classifier head (SVM and MLP, as in Stage 5/6).

This is the pretrained-embedding comparison/upper-bound requested by
faculty: same 31-way intent task, same speaker-grouped 5-fold CV
methodology established in Stage 7 (GroupKFold on speakerId, no per-fold
hyperparameter search), but the classifier now sits on top of mean-pooled
wav2vec2-base embeddings (768-dim, frozen, self-supervised on LibriSpeech --
never trained on FSC) instead of hand-built DSP features.

Hyperparameters are fixed across folds rather than grid-searched per fold
(unlike the single-split grid search in Stage 5/6): grid-searching inside
each of 5 folds would multiply the already-nontrivial SVM fit time on
~24k x 768 data by the grid size, and the point of this stage is the
features-vs-representation comparison, not squeezing out the last point of
accuracy from the classifier head.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

DATA_ROOT = "data/fluent_speech_commands_dataset"
N_SPLITS = 5


def main():
    data = np.load(f"{DATA_ROOT}/wav2vec2_embeddings.npz", allow_pickle=True)
    X, intent, speaker = data["X"], data["intent"], data["speakerId"]

    le = LabelEncoder().fit(intent)
    y = le.transform(intent)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    t_start = time.time()

    for fold, (train_idx, held_idx) in enumerate(gkf.split(X, y, groups=speaker)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        print(f"  train: {len(train_idx)} utterances, {len(set(speaker[train_idx]))} speakers")
        print(f"  held-out: {len(held_idx)} utterances, {len(set(speaker[held_idx]))} speakers")

        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_held = scaler.transform(X[held_idx])
        y_train, y_held = y[train_idx], y[held_idx]

        t0 = time.time()
        svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced").fit(X_train, y_train)
        svm_acc = (svm.predict(X_held) == y_held).mean()
        print(f"  SVM acc: {svm_acc:.4f} ({time.time()-t0:.0f}s)")

        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4,
                             max_iter=500, random_state=fold).fit(X_train, y_train)
        mlp_acc = (mlp.predict(X_held) == y_held).mean()
        print(f"  MLP acc: {mlp_acc:.4f} ({time.time()-t0:.0f}s)")

        fold_results.append({
            "fold": fold + 1, "svm_acc": svm_acc, "mlp_acc": mlp_acc,
            "n_train": len(train_idx), "n_held_out": len(held_idx),
        })
        pd.DataFrame(fold_results).to_csv(f"{DATA_ROOT}/wav2vec2_cv_results.csv", index=False)

    results_df = pd.DataFrame(fold_results)
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nSVM mean accuracy: {results_df['svm_acc'].mean():.4f} +/- {results_df['svm_acc'].std():.4f}")
    print(f"MLP mean accuracy: {results_df['mlp_acc'].mean():.4f} +/- {results_df['mlp_acc'].std():.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
