"""
Stage 8 classical baseline: one Gaussian HMM per intent class, trained on
that class's MFCC sequences, classifying a query utterance by the class
whose HMM assigns it the highest log-likelihood.

This is the classical, pre-deep-learning answer to sequence classification
(the same lineage as early ASR systems), kept as a same-footing comparison
point against the multi-head CNN (Stage 8) and the single-head CNN
(Stage 7) on identical features -- not a substitute for either.

Evaluated with the same speaker-grouped 5-fold cross-validation methodology
as Stages 7-8, for a directly comparable, equally reliable number.
"""

import time
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")  # hmmlearn convergence warnings on short/degenerate sequences

DATA_ROOT = "data/fluent_speech_commands_dataset"
N_SPLITS = 5
N_STATES = 3  # small, standard for short isolated-phrase HMMs


def true_length(seq):
    """Sequences are zero-padded to a fixed 350 frames; find the real (unpadded) length,
    since HMM training/scoring on the padding would drown out real per-class signal --
    typically only 40-65% of each padded sequence is real content."""
    nonzero = np.any(seq != 0, axis=1)
    return int(np.max(np.where(nonzero)[0]) + 1) if nonzero.any() else 1


def train_class_hmms(X_train, y_train, n_classes):
    """One diagonal-covariance GaussianHMM per class, trained on that class's sequences,
    each trimmed to its true (unpadded) length."""
    models = {}
    for c in range(n_classes):
        seqs = X_train[y_train == c]
        if len(seqs) < N_STATES:
            models[c] = None
            continue
        trimmed = [seq[:true_length(seq)] for seq in seqs]
        lengths = [len(t) for t in trimmed]
        concat = np.concatenate(trimmed, axis=0)
        model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                             n_iter=20, random_state=0)
        try:
            model.fit(concat, lengths)
            models[c] = model
        except Exception:
            models[c] = None
    return models


def classify(models, X, n_classes):
    preds = np.zeros(len(X), dtype=int)
    for i, seq in enumerate(X):
        trimmed_seq = seq[:true_length(seq)]
        best_c, best_score = -1, -np.inf
        for c in range(n_classes):
            m = models.get(c)
            if m is None:
                continue
            try:
                score = m.score(trimmed_seq)
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score, best_c = score, c
        preds[i] = best_c
    return preds


def main():
    data = np.load(f"{DATA_ROOT}/mfcc_sequences.npz", allow_pickle=True)
    X, intent, path = data["X"], data["intent"], data["path"]
    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    speaker_map = dict(zip(base["path"], base["speakerId"]))
    speaker = np.array([speaker_map[p] for p in path])

    le = LabelEncoder().fit(intent)
    y = le.transform(intent)
    n_classes = len(le.classes_)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    t_start = time.time()

    for fold, (train_idx, held_out_idx) in enumerate(gkf.split(X, y, groups=speaker)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        t_fold = time.time()
        X_train, y_train = X[train_idx], y[train_idx]
        X_held, y_held = X[held_out_idx], y[held_out_idx]

        models = train_class_hmms(X_train, y_train, n_classes)
        n_trained = sum(1 for m in models.values() if m is not None)
        print(f"  Trained {n_trained}/{n_classes} class HMMs")

        preds = classify(models, X_held, n_classes)
        acc = (preds == y_held).mean()
        print(f"  Fold {fold+1} accuracy: {acc:.4f} (fold time: {time.time()-t_fold:.0f}s)")

        fold_results.append({"fold": fold + 1, "accuracy": acc, "n_held_out": len(held_out_idx)})
        pd.DataFrame(fold_results).to_csv(f"{DATA_ROOT}/hmm_cv_results.csv", index=False)

    results_df = pd.DataFrame(fold_results)
    print("\n=== HMM BASELINE SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nMean accuracy: {results_df['accuracy'].mean():.4f} +/- {results_df['accuracy'].std():.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
