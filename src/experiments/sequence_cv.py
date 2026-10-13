"""
Speaker-grouped 5-fold cross-validation of the 1D-CNN for a chosen input
representation and prediction target.

Inputs:
  logmel        40-band log-mel sequences (as in cnn_logmel_cv.py)
  logmel_dct40  orthonormal DCT-II of the same log-mel frames, all 40
                coefficients kept (invertible: isolates decorrelation alone)
  logmel_dct13  same DCT, first 13 coefficients kept (adds MFCC-style
                truncation on top of decorrelation)
  wav2vec2      frozen wav2vec 2.0 frame sequences after PCA to 128 dims
Targets: intent (31 classes), action (6), object (14), location (4).

Architecture, fold assignment, 90/10 per-fold validation split, input
standardisation and training settings are identical to cnn_logmel_cv.py
and wav2vec2_sequence_cv.py, so results are directly comparable fold by fold.

Usage: python src/experiments/sequence_cv.py <input> <target>
"""

import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.fft import dct
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_ROOT, RESULTS_ROOT

N_SPLITS = 5


def load_input(name):
    """Returns X (N, T, C), labels table, speaker ids -- all in the same row order."""
    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv").set_index("path")
    if name == "wav2vec2":
        data = np.load(f"{DATA_ROOT}/wav2vec2_sequences.npz", allow_pickle=True)
    else:
        data = np.load(f"{DATA_ROOT}/logmel_sequences.npz", allow_pickle=True)
    X, path = data["X"], data["path"]
    if name.startswith("logmel_dct"):
        n_keep = int(name.replace("logmel_dct", ""))
        # Padded frames are all-zero, and the DCT of a zero frame is zero, so padding is preserved
        X = dct(X, type=2, norm="ortho", axis=-1)[:, :, :n_keep].astype(np.float32)
    labels = base.loc[path, ["action", "object", "location", "speakerId"]].reset_index(drop=True)
    labels["intent"] = labels["action"] + "_" + labels["object"] + "_" + labels["location"]
    return X, labels, labels["speakerId"].values


def build_model(input_shape, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
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
    input_name, target = sys.argv[1], sys.argv[2]
    X, labels, speaker = load_input(input_name)
    y = LabelEncoder().fit_transform(labels[target].values)
    n_classes = len(np.unique(y))
    print(f"Input {input_name} {X.shape}, target {target} ({n_classes} classes)", flush=True)

    prefix = {"logmel": "cnn_logmel", "logmel_dct40": "cnn_logmel_dct40", "logmel_dct13": "cnn_logmel_dct13",
              "wav2vec2": "wav2vec2_sequence"}[input_name]
    out_csv = f"{RESULTS_ROOT}/cv_{prefix}_{target}.csv"
    fold_results = []
    t_start = time.time()
    for fold, (trainval_idx, held_out_idx) in enumerate(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=speaker)):
        trainval_speakers = np.unique(speaker[trainval_idx])
        rng = np.random.RandomState(fold)
        rng.shuffle(trainval_speakers)
        val_speakers = set(trainval_speakers[:max(1, int(0.1 * len(trainval_speakers)))])
        train_idx = trainval_idx[[speaker[i] not in val_speakers for i in trainval_idx]]
        val_idx = trainval_idx[[speaker[i] in val_speakers for i in trainval_idx]]

        mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
        std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
        X_train, X_val, X_held = [(X[idx] - mean) / std for idx in (train_idx, val_idx, held_out_idx)]

        tf.random.set_seed(fold)
        model = build_model(X.shape[1:], n_classes)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
        t_fold = time.time()
        model.fit(X_train, y[train_idx], validation_data=(X_val, y[val_idx]),
                  batch_size=128, epochs=50, callbacks=[early_stop], verbose=0)
        held_pred = np.argmax(model.predict(X_held, verbose=0), axis=1)
        held_acc = (held_pred == y[held_out_idx]).mean()
        print(f"  fold {fold+1}: held-out acc {held_acc:.4f} ({time.time()-t_fold:.0f}s, "
              f"{len(model.history.history['loss'])} epochs)", flush=True)

        fold_results.append({"fold": fold + 1, "held_out_acc": held_acc, "n_classes": n_classes,
                             "n_train": len(train_idx), "n_val": len(val_idx), "n_held_out": len(held_out_idx)})
        pd.DataFrame(fold_results).to_csv(out_csv, index=False)
        del X_train, X_val, X_held, model
        tf.keras.backend.clear_session()

    df = pd.DataFrame(fold_results)
    print(f"Mean held-out accuracy: {df['held_out_acc'].mean():.4f} +/- {df['held_out_acc'].std():.4f} "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
