# Re-run of fold 1 of the wav2vec2-sequence CNN (identical protocol to
# src/experiments/wav2vec2_sequence_cv.py) that additionally saves held-out predictions.
import sys, numpy as np, tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from src.config import DATA_ROOT, RESULTS_ROOT
OUT = f"{RESULTS_ROOT}/wav2vec2_sequence_fold1_predictions.npz"
MAX_FRAMES, N_COMPONENTS = 175, 128
def build_model(n_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_FRAMES, N_COMPONENTS)),
        tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
data = np.load(f"{DATA_ROOT}/wav2vec2_sequences.npz", allow_pickle=True)
X, intent, speaker = data["X"], data["intent"], data["speakerId"]
le = LabelEncoder().fit(intent); y = le.transform(intent)
trainval_idx, held_idx = next(iter(GroupKFold(n_splits=5).split(X, y, groups=speaker)))
sp = np.unique(speaker[trainval_idx]); rng = np.random.RandomState(0); rng.shuffle(sp)
val_sp = set(sp[:max(1, int(0.1*len(sp)))])
tr = trainval_idx[[s not in val_sp for s in speaker[trainval_idx]]]
va = trainval_idx[[s in val_sp for s in speaker[trainval_idx]]]
mu = X[tr].mean(axis=(0,1), keepdims=True); sd = X[tr].std(axis=(0,1), keepdims=True)+1e-8
tf.random.set_seed(0)
m = build_model(len(le.classes_))
m.fit((X[tr]-mu)/sd, y[tr], validation_data=((X[va]-mu)/sd, y[va]), batch_size=128, epochs=50,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)], verbose=2)
pred = np.argmax(m.predict((X[held_idx]-mu)/sd, verbose=0), axis=1)
print("fold1 held-out acc", (pred == y[held_idx]).mean())
np.savez(OUT, y_true=y[held_idx], y_pred=pred, classes=le.classes_, speaker=speaker[held_idx])
