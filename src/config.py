"""Shared paths and analysis settings. All modules are run from the repository
root, e.g. `python -m src.features.utterance_prosodic`."""

DATA_ROOT = "data/fluent_speech_commands_dataset"   # FSC audio, metadata and cached features
RESULTS_ROOT = "results"                             # cross-validation results (tracked)
FIGURES_ROOT = "results/figures"                     # figures used in the report and slides

FEATURES_CSV = f"{DATA_ROOT}/features_cache.csv"     # utterance-level prosodic statistics
FRAME_LEN, HOP_LEN = 400, 160                        # 25 ms frames, 10 ms hop at 16 kHz
