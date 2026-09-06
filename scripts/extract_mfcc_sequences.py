"""
Stage 7 feature extraction: full per-frame MFCC sequences (not summarized to
mean/std), for a 1D-CNN that can learn from temporal order directly.

Same 25 ms frame / 10 ms hop / Hamming window / 13 coefficients as Stage 6.
Each utterance's (13, T) MFCC matrix is transposed to (T, 13) (time-major, as
Keras Conv1D expects) and padded/truncated to a fixed MAX_FRAMES so all
utterances share one array shape. MAX_FRAMES=350 covers roughly the 95th
percentile of utterance lengths in the corpus; longer utterances are
truncated, which discards trailing audio for the ~5% of clips beyond it.

Output is a single .npz (not CSV): faster to load, and appropriate for
storing a 3D array (utterances x time x coefficients) rather than a flat
feature table.
"""

import time
import numpy as np
import pandas as pd
import librosa

from extract_features import DATA_ROOT, FRAME_LEN, HOP_LEN

N_MFCC = 13
MAX_FRAMES = 350
SEQ_CACHE_PATH = f"{DATA_ROOT}/mfcc_sequences.npz"


def extract_mfcc_sequence(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=FRAME_LEN, hop_length=HOP_LEN, win_length=FRAME_LEN,
        window="hamming", center=False,
    )  # shape (N_MFCC, T)
    if mfcc.shape[1] < 1:
        return None
    seq = mfcc.T.astype(np.float32)  # (T, N_MFCC), time-major for Conv1D
    T = seq.shape[0]
    if T >= MAX_FRAMES:
        return seq[:MAX_FRAMES]
    padded = np.zeros((MAX_FRAMES, N_MFCC), dtype=np.float32)
    padded[:T] = seq
    return padded


def main():
    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    X, splits, intents, actions, paths = [], [], [], [], []
    t_start = time.time()
    n_done, n_failed = 0, 0

    for i, row in base.iterrows():
        wav_path = f"{DATA_ROOT}/{row['path']}"
        try:
            seq = extract_mfcc_sequence(wav_path)
        except Exception as e:
            print(f"  FAILED {wav_path}: {e}")
            n_failed += 1
            continue
        if seq is None:
            print(f"  SKIPPED (no frames) {wav_path}")
            n_failed += 1
            continue
        X.append(seq)
        splits.append(row["split"])
        intents.append(row["intent"])
        actions.append(row["action"])
        paths.append(row["path"])
        n_done += 1
        if n_done % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"  ...{n_done} files done, {elapsed/60:.1f} min elapsed, "
                  f"{elapsed/n_done*1000:.1f} ms/file")

    X = np.stack(X)  # (N, MAX_FRAMES, N_MFCC)
    np.savez_compressed(
        SEQ_CACHE_PATH,
        X=X,
        split=np.array(splits),
        intent=np.array(intents),
        action=np.array(actions),
        path=np.array(paths),
    )
    elapsed = time.time() - t_start
    print(f"Done. X shape={X.shape}, {n_failed} failed/skipped, "
          f"written to {SEQ_CACHE_PATH} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
