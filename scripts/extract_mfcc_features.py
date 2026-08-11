"""
Stage 6 feature extraction: adds mel-frequency cepstral coefficient (MFCC)
summary statistics to the Stage 5 prosodic feature cache.

MFCCs are computed with librosa.feature.mfcc using the same 25 ms frame
length, 10 ms hop, and Hamming window used throughout Stages 0-4 (center=False
to match the non-padded framing used elsewhere in the project). The first 13
coefficients are kept per frame and aggregated to mean/std across the
utterance (26 features), then merged with the existing features_cache.csv on
utterance path, reusing the Stage 5 prosodic features rather than
recomputing them.
"""

import time
import numpy as np
import pandas as pd
import librosa

from extract_features import DATA_ROOT, FRAME_LEN, HOP_LEN, CACHE_PATH

N_MFCC = 13
MFCC_CACHE_PATH = f"{DATA_ROOT}/features_cache_mfcc.csv"


def extract_mfcc_stats(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=FRAME_LEN, hop_length=HOP_LEN, win_length=FRAME_LEN,
        window="hamming", center=False,
    )
    if mfcc.shape[1] < 1:
        return None
    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    return feats


def run_mfcc_extraction():
    base = pd.read_csv(CACHE_PATH)
    rows = []
    t_start = time.time()

    for i, row in base.iterrows():
        wav_path = f"{DATA_ROOT}/{row['path']}"
        try:
            feats = extract_mfcc_stats(wav_path)
        except Exception as e:
            print(f"  FAILED {wav_path}: {e}")
            continue
        if feats is None:
            print(f"  SKIPPED (no frames) {wav_path}")
            continue
        feats["path"] = row["path"]
        rows.append(feats)
        if len(rows) % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"  ...{len(rows)} files done, {elapsed/60:.1f} min elapsed, "
                  f"{elapsed/len(rows)*1000:.1f} ms/file")

    mfcc_df = pd.DataFrame(rows)
    out = base.merge(mfcc_df, on="path", how="inner")
    out.to_csv(MFCC_CACHE_PATH, index=False)
    elapsed = time.time() - t_start
    print(f"Done. {len(out)} rows written to {MFCC_CACHE_PATH} in {elapsed/60:.1f} min "
          f"({elapsed/len(rows)*1000:.1f} ms/file)")
    return out


if __name__ == "__main__":
    run_mfcc_extraction()
