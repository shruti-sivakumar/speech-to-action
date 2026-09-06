# speech-to-action

Speech Processing course project: predicting spoken command intent from audio. The core pipeline (Stages 0-5) uses only classical speech-processing features (energy, zero-crossing rate, pitch, spectrogram, linear prediction) with no pretrained end-to-end speech model (no Whisper/wav2vec2/SLU black boxes) — the classifier never sees raw audio, only hand-built features. Later stages extend this with MFCCs, a from-scratch-trained sequence model, and classical/deep sequence-modeling comparisons, per faculty guidance (see `CLAUDE.md`).

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
| 5 | Feature vector + classifier | `notebooks/stage5_feature_classifier.ipynb` |
| 6 | MFCC features | `notebooks/stage6_mfcc_features.ipynb` |
| 7 | 1D-CNN over MFCC sequences (temporal order) | `notebooks/stage7_cnn_mfcc_sequence.ipynb` |
| 8 | Multi-head sequence model + DTW/HMM classical baselines | `notebooks/stage8_multihead_and_baselines.ipynb` |
| 9 | Pretrained-embedding comparison (wav2vec2) | `notebooks/stage9_pretrained_embedding_comparison.ipynb` |
| 10 | Final ablation ladder / write-up support | pending |

Each notebook is self-contained and executed, with plots and a Results section documenting the observed output. Stages 7-8 use `scripts/extract_mfcc_sequences.py` and `scripts/run_*_cv.py` / `scripts/run_*_baseline.py` for full-dataset feature extraction and cross-validated training, since these are too slow (minutes to tens of minutes) to run inline in a notebook cell — the notebooks load cached results.

## Environment

Python 3, virtual environment (`venv/`). Libraries: `numpy`, `scipy`, `matplotlib`, `librosa`, `scikit-learn`, `pandas`, `jupyter`, `tensorflow` (Stage 7-8 sequence models), `hmmlearn`, `fastdtw` (Stage 8 classical baselines), `torch`, `transformers` (Stage 9 frozen wav2vec2 embeddings). No GPU used or required — Stage 9 uses wav2vec2-base only as a frozen feature extractor (no fine-tuning), which is CPU-feasible.

## Methodology notes (Stage 7 onward)

The corpus's official validation and test splits contain only 10 speakers each. Stage 7 found this makes any single fixed-split accuracy unreliable for a high-capacity model (a single run showed 84% test vs. 62-67% validation accuracy, traced to a few unusually hard speakers dominating the validation split by chance). From Stage 7 onward, all deep and classical sequence models are evaluated with 5-fold **speaker-grouped** cross-validation (`GroupKFold` on speaker ID) instead of the fixed split, and results are reported as mean ± std across folds.
