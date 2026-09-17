"""
Stage 10 feature extraction: full per-frame log-mel spectrogram sequences.

This is the same recipe as Stage 7's MFCC sequences (`extract_mfcc_sequences.py`)
up to but not including the final DCT step of the MFCC pipeline: mel-filterbank,
then log-power compression, with no cepstral decorrelation. The point of the
stage is to test whether a CNN needs that decorrelation step at all, so every
other parameter (framing, hop, window, MAX_FRAMES) is held identical to Stage 7.

40 mel bands is used rather than librosa's internal MFCC default of 128 (Stage
6/7 never surface this number explicitly, since `librosa.feature.mfcc` computes
a 128-band mel spectrogram internally before DCT-truncating to 13 coefficients).
40 is the standard "log-mel filterbank" size used throughout the ASR literature
and keeps the cached array to ~1.7GB (30043 x 350 x 40 x 4 bytes) rather than
the ~5.4GB a 128-band version would need, which matters on an 8GB-RAM machine.

Output is a single .npz, same layout as mfcc_sequences.npz.
"""

import time
import numpy as np
import pandas as pd
import librosa

from extract_features import DATA_ROOT, FRAME_LEN, HOP_LEN

N_MELS = 40
MAX_FRAMES = 350
SEQ_CACHE_PATH = f"{DATA_ROOT}/logmel_sequences.npz"


def extract_logmel_sequence(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=FRAME_LEN, hop_length=HOP_LEN, win_length=FRAME_LEN,
        window="hamming", center=False,
    )  # shape (N_MELS, T), power spectrum
    if mel.shape[1] < 1:
        return None
    log_mel = librosa.power_to_db(mel)
    seq = log_mel.T.astype(np.float32)  # (T, N_MELS), time-major for Conv1D
    T = seq.shape[0]
    if T >= MAX_FRAMES:
        return seq[:MAX_FRAMES]
    padded = np.zeros((MAX_FRAMES, N_MELS), dtype=np.float32)
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
            seq = extract_logmel_sequence(wav_path)
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

    X = np.stack(X)  # (N, MAX_FRAMES, N_MELS)
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
