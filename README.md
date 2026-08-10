# speech-to-action

Speech Processing course project: predicting spoken command intent from audio using classical speech-processing features, without a pretrained end-to-end speech model (no Whisper/wav2vec2/SLU black boxes). The classifier only ever sees features computed by hand-built signal-processing stages (energy, zero-crossing rate, pitch, spectrogram, linear prediction).

## Dataset

[Fluent Speech Commands (FSC)](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus) — ~30,043 utterances, 97 speakers, 31 unique intents (action/object/location slots). Train/valid/test split is speaker-independent (verified: zero speaker overlap across splits). Dataset is placed locally under `data/fluent_speech_commands_dataset/`; audio (`wavs/`) is gitignored, CSV metadata is tracked. License is academic/research use only.

## Pipeline stages

| Stage | Description | Notebook |
|---|---|---|
| 0 | Load, frame, window | `notebooks/stage0_load_frame_window.ipynb` |
| 1 | Short-Time Energy, Zero-Crossing Rate, Voiced/Unvoiced/Silence | `notebooks/stage1_ste_zcr.ipynb` |
| 2 | Pitch (F0) via autocorrelation | `notebooks/stage2_pitch_autocorrelation.ipynb` |
| 3 | STFT / spectrogram | `notebooks/stage3_stft_spectrogram.ipynb` |
| 4 | Linear Prediction (LPC) | `notebooks/stage4_lpc.ipynb` |
| 5 | Feature vector + classifier | pending |

Each notebook is self-contained and executed, with plots and a Results section documenting the observed output.

## Environment

Python 3, virtual environment (`venv/`). Libraries: `numpy`, `scipy`, `matplotlib`, `librosa`, `scikit-learn`, `pandas`, `jupyter`.
