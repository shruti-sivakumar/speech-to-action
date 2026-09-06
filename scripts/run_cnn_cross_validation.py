"""
Stage 7 speaker-grouped k-fold cross-validation.

Motivated by a large, reproducible gap between the fixed-split validation
accuracy (~66%) and test accuracy (~82-83%) for the 1D-CNN. Per-speaker
analysis showed this is driven by only 10 speakers each in the original
valid/test splits: validation happened to contain two unusually difficult
speakers (0.23, 0.28 accuracy) dragging its average down, while test's 10
speakers were all comparatively easy (0.71-0.97). A single fixed split's
accuracy is therefore not a reliable estimate of generalization with this
few speakers per split.

This script pools all 97 speakers and all 30,043 utterances, and runs
GroupKFold (grouped by speakerId, so no speaker appears in both the training
and held-out side of any fold) with k=5. Each fold further splits its
training speakers 90/10 (by speaker, seeded per fold) for early-stopping
validation. The same architecture and training procedure as the main Stage 7
run is used throughout, so fold results are comparable to it.
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = "data/fluent_speech_commands_dataset"
N_SPLITS = 5


def build_model(n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(350, 13)),
        tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


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

    for fold, (trainval_idx, held_out_idx) in enumerate(gkf.split(X, y, groups=speaker)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        trainval_speakers = np.unique(speaker[trainval_idx])
        rng = np.random.RandomState(fold)
        rng.shuffle(trainval_speakers)
        n_val_speakers = max(1, int(0.1 * len(trainval_speakers)))
        val_speakers = set(trainval_speakers[:n_val_speakers])

        train_idx = trainval_idx[[speaker[i] not in val_speakers for i in trainval_idx]]
        val_idx = trainval_idx[[speaker[i] in val_speakers for i in trainval_idx]]

        print(f"  train: {len(train_idx)} utterances, {len(set(speaker[train_idx]))} speakers")
        print(f"  val:   {len(val_idx)} utterances, {len(set(speaker[val_idx]))} speakers")
        print(f"  held-out: {len(held_out_idx)} utterances, {len(set(speaker[held_out_idx]))} speakers")

        mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
        std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X[train_idx] - mean) / std
        X_val = (X[val_idx] - mean) / std
        X_held = (X[held_out_idx] - mean) / std
        y_train, y_val, y_held = y[train_idx], y[val_idx], y[held_out_idx]

        tf.random.set_seed(fold)
        model = build_model(n_classes)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True)
        t_fold = time.time()
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  batch_size=128, epochs=50, callbacks=[early_stop], verbose=2)

        held_pred = np.argmax(model.predict(X_held, verbose=0), axis=1)
        held_acc = (held_pred == y_held).mean()
        print(f"  Fold {fold+1} held-out accuracy: {held_acc:.4f} (fold time: {time.time()-t_fold:.0f}s)")

        fold_results.append({
            "fold": fold + 1, "held_out_acc": held_acc,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_held_out": len(held_out_idx),
        })
        pd.DataFrame(fold_results).to_csv(f"{DATA_ROOT}/cnn_cv_results.csv", index=False)

    results_df = pd.DataFrame(fold_results)
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nMean held-out accuracy: {results_df['held_out_acc'].mean():.4f} "
          f"+/- {results_df['held_out_acc'].std():.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
