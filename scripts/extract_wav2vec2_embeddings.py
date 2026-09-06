"""
Stage 9 feature extraction: frozen wav2vec2-base embeddings, one per utterance.

wav2vec2-base (facebook/wav2vec2-base) is self-supervised on 960h of
LibriSpeech (read English speech) -- no labels, and never trained on FSC.
It is used here purely as a frozen feature extractor: no fine-tuning, no
gradient ever flows into the encoder. This mirrors the Stage 5/6 pattern
(fixed feature computation -> small classifier on top) but swaps hand-built
DSP features for a pretrained model's learned representation, so the two
are a fair "features vs learned representation" comparison rather than a
"more training" comparison.

The encoder outputs one 768-dim vector per ~20ms frame (its own internal
hop, not the 10ms hop used elsewhere in this project). We mean-pool over
time to get a single fixed-length 768-dim vector per utterance -- the
standard "probing" setup for frozen speech representations.

Single-clip inference (not batched) is used: batching requires padding
shorter clips to the longest in the batch, which wastes compute given
FSC's variable utterance lengths, and was empirically *slower* per-clip
than a plain loop in a smoke test on this dataset/hardware.
"""

import time
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from extract_features import DATA_ROOT

EMB_CACHE_PATH = f"{DATA_ROOT}/wav2vec2_embeddings.npz"
MODEL_NAME = "facebook/wav2vec2-base"


def main():
    print(f"Loading {MODEL_NAME} (frozen, eval mode)...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    torch.set_grad_enabled(False)  # never need gradients for a frozen extractor

    base = pd.read_csv(f"{DATA_ROOT}/features_cache.csv")
    embeddings, splits, intents, actions, objects, locations = [], [], [], [], [], []
    speaker_ids, paths = [], []

    t_start = time.time()
    n_done, n_failed = 0, 0

    for i, row in base.iterrows():
        wav_path = f"{DATA_ROOT}/{row['path']}"
        try:
            y, sr = librosa.load(wav_path, sr=16000)
            inputs = processor(y, sampling_rate=16000, return_tensors="pt")
            out = model(**inputs)
            # last_hidden_state: (1, T, 768) -> mean-pool over time -> (768,)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        except Exception as e:
            print(f"  FAILED {wav_path}: {e}")
            n_failed += 1
            continue

        embeddings.append(emb.astype(np.float32))
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

    X = np.stack(embeddings)  # (N, 768)
    np.savez_compressed(
        EMB_CACHE_PATH,
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
          f"written to {EMB_CACHE_PATH} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
