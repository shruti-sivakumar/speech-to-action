"""
Stage 8 classical baseline: DTW (Dynamic Time Warping) k-NN classification.

DTW measures similarity between two variable-length sequences by finding
the lowest-cost alignment (warping) between them, rather than requiring a
fixed-length representation -- the classical, pre-deep-learning approach to
comparing sequences of different lengths (its original application domain
was isolated-word recognition). A query is classified by finding its
nearest template sequence(s) by DTW distance.

Full pairwise DTW (query x every training utterance) is computationally
infeasible here: fastdtw over one pair of our (350, 13) MFCC sequences
takes ~40ms, and there are ~23k training utterances per fold. Two scope
reductions are made and reported honestly: (1) classification is against a
small, fixed number of template sequences per class rather than the full
training set, a standard simplification for DTW-based classifiers; (2)
evaluation uses a stratified subset of one fold's held-out speakers rather
than the full 5-fold sweep used for the deep models, since DTW's per-query
cost is roughly 1000x a deep model's forward pass.
"""

import time
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = "data/fluent_speech_commands_dataset"
N_TEMPLATES_PER_CLASS = 5
N_QUERIES_PER_CLASS = 5
K_NEIGHBORS = 3


def true_length(seq):
    """Sequences are zero-padded to a fixed 350 frames; trim to the real (unpadded)
    length before computing DTW distance, for consistency with the HMM baseline fix
    (typically only 40-65% of each padded sequence is real content)."""
    nonzero = np.any(seq != 0, axis=1)
    return int(np.max(np.where(nonzero)[0]) + 1) if nonzero.any() else 1


def main():
    data = np.load(f"{DATA_ROOT}/mfcc_sequences.npz", allow_pickle=True)
    X, intent, path = data["X"], data["intent"], data["path"]
    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    speaker_map = dict(zip(base["path"], base["speakerId"]))
    speaker = np.array([speaker_map[p] for p in path])

    le = LabelEncoder().fit(intent)
    y = le.transform(intent)
    n_classes = len(le.classes_)

    gkf = GroupKFold(n_splits=5)
    train_idx, held_out_idx = next(gkf.split(X, y, groups=speaker))  # fold 1 only, for tractability
    print(f"Fold 1 split: train={len(train_idx)}, held_out={len(held_out_idx)}")

    rng = np.random.RandomState(0)
    template_idx, query_idx = [], []
    for c in range(n_classes):
        train_c = train_idx[y[train_idx] == c]
        held_c = held_out_idx[y[held_out_idx] == c]
        if len(train_c) >= N_TEMPLATES_PER_CLASS:
            template_idx.extend(rng.choice(train_c, N_TEMPLATES_PER_CLASS, replace=False))
        if len(held_c) >= N_QUERIES_PER_CLASS:
            query_idx.extend(rng.choice(held_c, N_QUERIES_PER_CLASS, replace=False))
        elif len(held_c) > 0:
            query_idx.extend(held_c)

    template_idx, query_idx = np.array(template_idx), np.array(query_idx)
    print(f"Templates: {len(template_idx)} ({N_TEMPLATES_PER_CLASS}/class), "
          f"Queries: {len(query_idx)} (up to {N_QUERIES_PER_CLASS}/class)")
    print(f"Estimated DTW calls: {len(template_idx) * len(query_idx)}")

    X_templates_trimmed = {ti: X[ti][:true_length(X[ti])] for ti in template_idx}

    t_start = time.time()
    preds = []
    for qi, q in enumerate(query_idx):
        q_trimmed = X[q][:true_length(X[q])]
        dists = []
        for ti in template_idx:
            dist, _ = fastdtw(q_trimmed, X_templates_trimmed[ti], dist=euclidean)
            dists.append(dist)
        dists = np.array(dists)
        nearest = np.argsort(dists)[:K_NEIGHBORS]
        nearest_labels = y[template_idx[nearest]]
        pred = np.bincount(nearest_labels, minlength=n_classes).argmax()
        preds.append(pred)
        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  ...{qi+1}/{len(query_idx)} queries, {elapsed/60:.1f} min elapsed, "
                  f"{elapsed/(qi+1):.2f} s/query")

    preds = np.array(preds)
    y_query = y[query_idx]
    acc = (preds == y_query).mean()
    elapsed = time.time() - t_start
    print(f"\nDTW k-NN (k={K_NEIGHBORS}) accuracy on {len(query_idx)} stratified queries: {acc:.4f}")
    print(f"Total time: {elapsed/60:.1f} min")

    pd.DataFrame([{"accuracy": acc, "n_queries": len(query_idx), "n_templates": len(template_idx),
                    "k": K_NEIGHBORS}]).to_csv(f"{DATA_ROOT}/dtw_baseline_result.csv", index=False)


if __name__ == "__main__":
    main()
