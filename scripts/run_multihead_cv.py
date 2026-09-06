"""
Stage 8: multi-head sequence model (action/object/location) with data
augmentation, evaluated via the same speaker-grouped 5-fold cross-validation
methodology established in Stage 7 (a fixed split is unreliable with only
10 speakers per side).

Architecture: the same convolutional trunk as Stage 7 (three Conv1D blocks),
but with three separate softmax output heads instead of one flat 31-way
head, predicting the action (6 classes), object (14 classes), and location
(4 classes) slots independently from one shared representation. "Full
intent accuracy" (all three heads correct simultaneously) is the number
comparable to Stage 7's flat accuracy.

Data augmentation: SpecAugment-style time and frequency masking, applied as
a custom Keras layer that is a no-op at inference time. This is standard,
well-established classical augmentation for spectral/cepstral sequence
features (not a black-box addition), and addresses the ~23k-utterance
training set being thin for training a multi-output model from scratch.
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = "data/fluent_speech_commands_dataset"
N_SPLITS = 5


class SpecAugment(tf.keras.layers.Layer):
    """Randomly zeroes contiguous time and coefficient blocks during training only."""

    def __init__(self, n_time_masks=2, time_mask_width=20, n_freq_masks=2, freq_mask_width=3, **kwargs):
        super().__init__(**kwargs)
        self.n_time_masks = n_time_masks
        self.time_mask_width = time_mask_width
        self.n_freq_masks = n_freq_masks
        self.freq_mask_width = freq_mask_width

    def call(self, inputs, training=None):
        if not training:
            return inputs
        shape = tf.shape(inputs)
        batch, T, C = shape[0], shape[1], shape[2]
        mask = tf.ones((batch, T, C), dtype=inputs.dtype)

        time_idx = tf.range(T)[tf.newaxis, :, tf.newaxis]
        for _ in range(self.n_time_masks):
            t0 = tf.random.uniform((batch, 1, 1), 0, T - self.time_mask_width, dtype=tf.int32)
            time_mask = tf.cast((time_idx < t0) | (time_idx >= t0 + self.time_mask_width), inputs.dtype)
            mask = mask * time_mask

        freq_idx = tf.range(C)[tf.newaxis, tf.newaxis, :]
        for _ in range(self.n_freq_masks):
            f0 = tf.random.uniform((batch, 1, 1), 0, C - self.freq_mask_width, dtype=tf.int32)
            freq_mask = tf.cast((freq_idx < f0) | (freq_idx >= f0 + self.freq_mask_width), inputs.dtype)
            mask = mask * freq_mask

        return inputs * mask


def build_model(n_action, n_object, n_location):
    inputs = tf.keras.Input(shape=(350, 13))
    x = SpecAugment()(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    shared = tf.keras.layers.Dense(128, activation="relu")(x)
    shared = tf.keras.layers.Dropout(0.3)(shared)

    action_out = tf.keras.layers.Dense(n_action, activation="softmax", name="action")(shared)
    object_out = tf.keras.layers.Dense(n_object, activation="softmax", name="object")(shared)
    location_out = tf.keras.layers.Dense(n_location, activation="softmax", name="location")(shared)

    model = tf.keras.Model(inputs, [action_out, object_out, location_out])
    model.compile(
        optimizer="adam",
        loss={"action": "sparse_categorical_crossentropy",
              "object": "sparse_categorical_crossentropy",
              "location": "sparse_categorical_crossentropy"},
        metrics={"action": "accuracy", "object": "accuracy", "location": "accuracy"},
    )
    return model


def main():
    data = np.load(f"{DATA_ROOT}/mfcc_sequences.npz", allow_pickle=True)
    X, path = data["X"], data["path"]

    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    meta = base.set_index("path").loc[path]
    speaker = meta["speakerId"].values
    action_enc = LabelEncoder().fit(meta["action"])
    object_enc = LabelEncoder().fit(meta["object"])
    location_enc = LabelEncoder().fit(meta["location"])
    y_action = action_enc.transform(meta["action"])
    y_object = object_enc.transform(meta["object"])
    y_location = location_enc.transform(meta["location"])
    n_action, n_object, n_location = len(action_enc.classes_), len(object_enc.classes_), len(location_enc.classes_)
    print(f"n_action={n_action}, n_object={n_object}, n_location={n_location}")

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []
    t_start = time.time()

    for fold, (trainval_idx, held_out_idx) in enumerate(gkf.split(X, groups=speaker)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
        trainval_speakers = np.unique(speaker[trainval_idx])
        rng = np.random.RandomState(fold)
        rng.shuffle(trainval_speakers)
        n_val_speakers = max(1, int(0.1 * len(trainval_speakers)))
        val_speakers = set(trainval_speakers[:n_val_speakers])

        train_idx = trainval_idx[[speaker[i] not in val_speakers for i in trainval_idx]]
        val_idx = trainval_idx[[speaker[i] in val_speakers for i in trainval_idx]]
        print(f"  train: {len(train_idx)}, val: {len(val_idx)}, held-out: {len(held_out_idx)}")

        mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
        std = X[train_idx].std(axis=(0, 1), keepdims=True) + 1e-8
        X_train, X_val, X_held = (X[train_idx] - mean) / std, (X[val_idx] - mean) / std, (X[held_out_idx] - mean) / std

        y_train = {"action": y_action[train_idx], "object": y_object[train_idx], "location": y_location[train_idx]}
        y_val = {"action": y_action[val_idx], "object": y_object[val_idx], "location": y_location[val_idx]}

        tf.random.set_seed(fold)
        model = build_model(n_action, n_object, n_location)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        t_fold = time.time()
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  batch_size=128, epochs=50, callbacks=[early_stop], verbose=2)

        preds = model.predict(X_held, verbose=0)
        action_pred = np.argmax(preds[0], axis=1)
        object_pred = np.argmax(preds[1], axis=1)
        location_pred = np.argmax(preds[2], axis=1)

        y_action_held, y_object_held, y_location_held = y_action[held_out_idx], y_object[held_out_idx], y_location[held_out_idx]
        action_acc = (action_pred == y_action_held).mean()
        object_acc = (object_pred == y_object_held).mean()
        location_acc = (location_pred == y_location_held).mean()
        full_acc = ((action_pred == y_action_held) & (object_pred == y_object_held) & (location_pred == y_location_held)).mean()

        print(f"  Fold {fold+1}: action_acc={action_acc:.4f} object_acc={object_acc:.4f} "
              f"location_acc={location_acc:.4f} FULL_INTENT_acc={full_acc:.4f} (fold time: {time.time()-t_fold:.0f}s)")

        fold_results.append({
            "fold": fold + 1, "action_acc": action_acc, "object_acc": object_acc,
            "location_acc": location_acc, "full_intent_acc": full_acc,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_held_out": len(held_out_idx),
        })
        pd.DataFrame(fold_results).to_csv(f"{DATA_ROOT}/multihead_cv_results.csv", index=False)

    results_df = pd.DataFrame(fold_results)
    print("\n=== MULTI-HEAD CROSS-VALIDATION SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nMean full-intent accuracy: {results_df['full_intent_acc'].mean():.4f} "
          f"+/- {results_df['full_intent_acc'].std():.4f}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
