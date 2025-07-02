# EEG_preprocessing

This folder gathers all the utilities to prepare the EEG signals before model training.
Below is a short description of each script.

| Script | Description |
| ------ | ----------- |
| `segment_raw_signals_200Hz.py` | Splits raw SEED-DV recordings (62 channels, 200 Hz) into 2-second windows organized as `(block, concept, repetition, channel, time)`.
| `segment_sliding_window.py` | Further divides the 2-second windows into 500 ms windows with 250 ms overlap, producing a `(block, concept, repetition, window, channel, time)` tensor.
| `extract_DE_PSD_features_1per2s.py` | Computes one Differential Entropy and one Power Spectral Density feature per 2-second segment over five frequency bands.
| `extract_DE_PSD_features_1per1s.py` | Similar to the previous script but uses two non-overlapping 1-second windows inside each 2-second segment.
| `extract_DE_PSD_features_1per500ms.py` | Extracts DE and PSD features for every 500 ms sliding window.
| `DE_PSD.py` | Implements the actual DE and PSD computations from an EEG segment.
| `__init__.py` | Marks this folder as a Python package.
