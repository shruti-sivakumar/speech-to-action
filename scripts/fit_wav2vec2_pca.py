"""
Stage 9b, pass A: fit a PCA projection (768 -> N_COMPONENTS) on a sample of
wav2vec2-base frame embeddings, so the full per-frame sequence extraction
(pass B, extract_wav2vec2_sequences.py) can store a much smaller array.

Why this is needed: caching full (T, 768) sequences for all ~30k utterances
at float32 would be roughly 15GB (175 frames x 768 dims x 30043 utterances x
4 bytes), close to or exceeding this machine's 8GB RAM even before training
starts. Reducing to 128 dims/frame via PCA cuts that to ~2.5GB, comfortably
within budget, while (per the whole point of PCA) keeping the directions of
highest variance in the original representation.

Fit only on frames from the *train* split, matching standard practice for
any data-dependent preprocessing step (avoids the projection being shaped
even slightly by validation/test/held-out data) -- fitting is unsupervised
(no labels used), but the utterances themselves should still come only from
train.
"""

import time
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
import joblib

from extract_features import DATA_ROOT

N_COMPONENTS = 128
TARGET_FRAMES = 300_000  # stop sampling once we have roughly this many frames
PCA_PATH = f"{DATA_ROOT}/wav2vec2_pca.joblib"
MODEL_NAME = "facebook/wav2vec2-base"


def main():
    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    torch.set_grad_enabled(False)

    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    train_rows = base[base["split"] == "train"].sample(frac=1, random_state=0)

    frames = []
    n_frames = 0
    n_utts = 0
    t_start = time.time()

    for _, row in train_rows.iterrows():
        if n_frames >= TARGET_FRAMES:
            break
        wav_path = f"{DATA_ROOT}/{row['path']}"
        y, sr = librosa.load(wav_path, sr=16000)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        out = model(**inputs)
        emb = out.last_hidden_state.squeeze(0).numpy()  # (T, 768)
        frames.append(emb)
        n_frames += emb.shape[0]
        n_utts += 1
        if n_utts % 500 == 0:
            print(f"  ...{n_utts} utterances, {n_frames} frames collected, "
                  f"{(time.time()-t_start)/60:.1f} min elapsed")

    X = np.concatenate(frames, axis=0)  # (n_frames, 768)
    print(f"Fitting PCA: {X.shape} -> {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
    pca.fit(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio (sum of top {N_COMPONENTS} components): {explained:.4f}")

    joblib.dump(pca, PCA_PATH)
    print(f"Saved PCA to {PCA_PATH}. Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
