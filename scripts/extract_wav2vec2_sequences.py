"""
Stage 9b, pass B: full per-frame wav2vec2-base embedding sequences (PCA-
reduced to 128 dims/frame), for a Stage 7/8-style 1D-CNN that can exploit
the pretrained representation's temporal structure directly, rather than
Stage 9's mean-pooled static vector.

Requires fit_wav2vec2_pca.py to have been run first (produces
wav2vec2_pca.joblib). Each utterance's (T, 768) wav2vec2 output is
projected to (T, 128) via the fitted PCA, then padded/truncated to
MAX_FRAMES=175 -- chosen the same way Stage 7 chose MAX_FRAMES=350 for
MFCC: it covers the corpus's ~95th percentile utterance duration (3.41s),
verified empirically to correspond to ~170 wav2vec2 frames (measured
directly, not assumed from the nominal 20ms/frame architecture spec).
"""

import time
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib

from extract_features import DATA_ROOT

N_COMPONENTS = 128
MAX_FRAMES = 175
PCA_PATH = f"{DATA_ROOT}/wav2vec2_pca.joblib"
SEQ_CACHE_PATH = f"{DATA_ROOT}/wav2vec2_sequences.npz"
MODEL_NAME = "facebook/wav2vec2-base"


def extract_sequence(y, processor, model, pca):
    inputs = processor(y, sampling_rate=16000, return_tensors="pt")
    out = model(**inputs)
    emb = out.last_hidden_state.squeeze(0).numpy()  # (T, 768)
    reduced = pca.transform(emb).astype(np.float32)  # (T, 128)
    T = reduced.shape[0]
    if T >= MAX_FRAMES:
        return reduced[:MAX_FRAMES]
    padded = np.zeros((MAX_FRAMES, N_COMPONENTS), dtype=np.float32)
    padded[:T] = reduced
    return padded


def main():
    print(f"Loading {MODEL_NAME} and PCA projection...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    torch.set_grad_enabled(False)
    pca = joblib.load(PCA_PATH)

    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    X, splits, intents, actions, objects, locations = [], [], [], [], [], []
    speaker_ids, paths = [], []

    t_start = time.time()
    n_done, n_failed = 0, 0

    for i, row in base.iterrows():
        wav_path = f"{DATA_ROOT}/{row['path']}"
        try:
            y, sr = librosa.load(wav_path, sr=16000)
            seq = extract_sequence(y, processor, model, pca)
        except Exception as e:
            print(f"  FAILED {wav_path}: {e}")
            n_failed += 1
            continue

        X.append(seq)
        splits.append(row["split"])
        intents.append(row["intent"])
        actions.append(row["action"])
        objects.append(row["object"])
        locations.append(row["location"])
        speaker_ids.append(row["speakerId"])
        paths.append(row["path"])
        n_done += 1
        if n_done % 2000 == 0:
            elapsed = time.time() - t_start
            print(f"  ...{n_done} files done, {elapsed/60:.1f} min elapsed, "
                  f"{elapsed/n_done*1000:.1f} ms/file")

    X = np.stack(X)  # (N, MAX_FRAMES, N_COMPONENTS)
    np.savez_compressed(
        SEQ_CACHE_PATH,
        X=X,
        split=np.array(splits),
        intent=np.array(intents),
        action=np.array(actions),
        object=np.array(objects),
        location=np.array(locations),
        speakerId=np.array(speaker_ids),
        path=np.array(paths),
    )
    elapsed = time.time() - t_start
    print(f"Done. X shape={X.shape}, {n_failed} failed, "
          f"written to {SEQ_CACHE_PATH} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
