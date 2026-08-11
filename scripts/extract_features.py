"""
Feature extraction over the FSC dataset for Stage 5.

Collapses each utterance's per-frame features (energy, ZCR, voicing, pitch,
formants) to a fixed-length summary vector and caches the result to CSV, so
the Stage 5 notebook can load it instantly instead of recomputing ~30k
files' worth of features on every run.

Pitch is estimated with librosa.yin rather than the from-scratch autocorrelation
method of Stage 2 or librosa.pyin, purely for runtime at full-dataset scale
(pyin is roughly 70x slower per file). Formants are averaged over every frame
classified voiced by the Stage-1-style thresholding below, using librosa.lpc
(Burg's method).
"""

import time
import numpy as np
import pandas as pd
import librosa
from scipy.signal import freqz, find_peaks

DATA_ROOT = "data/fluent_speech_commands_dataset"
FRAME_LEN, HOP_LEN = 400, 160
LPC_ORDER = 16
CACHE_PATH = f"{DATA_ROOT}/features_cache.csv"


def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    frames = librosa.util.frame(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
    n_frames = frames.shape[1]
    if n_frames < 1:
        return None

    ste = np.sum(frames.astype(np.float64) ** 2, axis=0)
    ste_db = 10 * np.log10(ste + 1e-10)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs, axis=0)) / 2, axis=0)

    noise_floor_db = np.percentile(ste_db, 15)
    is_speech = ste_db > (noise_floor_db + 10)
    if is_speech.sum() > 0:
        zcr_thresh = np.percentile(zcr[is_speech], 70)
        voiced = is_speech & (zcr <= zcr_thresh)
    else:
        voiced = np.zeros(n_frames, dtype=bool)

    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=FRAME_LEN * 2, hop_length=HOP_LEN)
    f0 = f0[:n_frames]
    f0_voiced = f0[voiced] if voiced.sum() > 0 else np.array([0.0])

    # F1/F2 tracked separately per frame (lowest two formant peaks), not pooled together,
    # since blending F1 and F2 into one averaged number would discard the distinction
    # between them entirely.
    formant_idxs = np.where(voiced)[0] if voiced.sum() > 0 else np.argsort(-ste)[:3]

    f1_estimates, f2_estimates = [], []
    win = np.hamming(FRAME_LEN)
    for idx in formant_idxs:
        frame = frames[:, idx].astype(np.float64) * win
        if frame.std() == 0:
            continue
        try:
            a = librosa.lpc(frame, order=LPC_ORDER)
        except Exception:
            continue
        ww, hh = freqz([1.0], a, worN=512, fs=sr)
        mag = 20 * np.log10(np.abs(hh) + 1e-10)
        mask = ww < 3500
        pk, _ = find_peaks(mag[mask], prominence=0.3)
        peak_freqs = ww[mask][pk]
        if len(peak_freqs) > 0:
            f1_estimates.append(peak_freqs[0])
        if len(peak_freqs) > 1:
            f2_estimates.append(peak_freqs[1])

    return {
        "mean_energy_db": np.mean(ste_db),
        "std_energy_db": np.std(ste_db),
        "mean_zcr": np.mean(zcr),
        "std_zcr": np.std(zcr),
        "voiced_fraction": voiced.mean(),
        "mean_f0": np.mean(f0_voiced),
        "std_f0": np.std(f0_voiced),
        "mean_f1": np.mean(f1_estimates) if f1_estimates else 0.0,
        "mean_f2": np.mean(f2_estimates) if f2_estimates else 0.0,
        "duration_s": len(y) / sr,
    }


def run_extraction(cache_path=CACHE_PATH):
    splits = {"train": "train_data.csv", "valid": "valid_data.csv", "test": "test_data.csv"}
    rows = []
    t_start = time.time()
    total_done = 0

    for split, csv_name in splits.items():
        df = pd.read_csv(f"{DATA_ROOT}/data/{csv_name}")
        print(f"[{split}] {len(df)} rows")
        for i, row in df.iterrows():
            wav_path = f"{DATA_ROOT}/{row['path']}"
            try:
                feats = extract_features(wav_path)
            except Exception as e:
                print(f"  FAILED {wav_path}: {e}")
                continue
            if feats is None:
                print(f"  SKIPPED (no frames) {wav_path}")
                continue
            feats.update({
                "split": split,
                "path": row["path"],
                "speakerId": row["speakerId"],
                "action": row["action"],
                "object": row["object"],
                "location": row["location"],
                "intent": f"{row['action']}_{row['object']}_{row['location']}",
            })
            rows.append(feats)
            total_done += 1
            if total_done % 2000 == 0:
                elapsed = time.time() - t_start
                print(f"  ...{total_done} files done, {elapsed/60:.1f} min elapsed, "
                      f"{elapsed/total_done*1000:.1f} ms/file")

    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    elapsed = time.time() - t_start
    print(f"Done. {len(out)} rows written to {cache_path} in {elapsed/60:.1f} min")
    print(f"Unique intents: {out['intent'].nunique()}")
    return out


if __name__ == "__main__":
    run_extraction()
