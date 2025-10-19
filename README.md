# Brain-Controlled Smart Glasses - EEG Command Classifier

End-to-end ML pipeline to classify EEG signals into 5 mental commands: neutral, fan_on, fan_off, light_on, light_off. Supports both a Random Forest (features) and a 1D CNN (raw signals) model, with synthetic data generation and real-time inference demos.

## Project structure
- `config.py` — global params and repo-relative paths (`data/`, `models/`, `results/`).
- `generate_synthetic_eeg.py` — create labeled, EEG-like synthetic data.
- `preprocess_data.py` — bandpass 8–30 Hz, artifact clipping, z-score normalization.
- `extract_features.py` — time/frequency features per sample.
- `split_data.py` — stratified split; writes train/val for both signals and features.
- `train_random_forest.py` — baseline training on features; saves `.pkl` and metrics.
- `train_cnn.py` — 1D CNN training on signals; saves `.pth` and metrics/plots.
- `inference.py` — Random Forest inference engine.
- `inference_cnn.py` — CNN inference engine (single/batch, with warmup).
- `demo_inference.py` — side-by-side RF vs CNN demo + visualizations.
- `visualize_predictions.py` — plotting utilities.

## Installation
1) Create/activate a Python 3.9+ env.
2) Install dependencies:
```
pip install -r requirements.txt
```

## End-to-end pipeline (synthetic data)
From the project root (same folder as this README):

1) Generate data:
```
python generate_synthetic_eeg.py
```
2) Preprocess (bandpass 8–30 Hz, normalize):
```
python preprocess_data.py
```
3) Extract features:
```
python extract_features.py
```
4) Split train/val (stratified, shared indices):
```
python split_data.py
```
5) Train baseline (Random Forest):
```
python train_random_forest.py
```
6) Train 1D CNN:
```
python train_cnn.py
```

## Inference demos
- RF single-sample inference test:
```
python inference.py
```
- CNN single/batch inference test:
```
python inference_cnn.py
```
- Compare both + visualization:
```
python demo_inference.py
```

Artifacts are written to `models/` and `results/`.

## Notes
- All paths are repo-relative (Windows/macOS/Linux friendly).
- If you see DataLoader issues on Windows, set `num_workers=0` in `train_cnn.py`.
- To use a public dataset, adapt loaders to produce arrays matching:
  - Signals: `(n_samples, n_channels, n_timepoints)` (500 timepoints for 2s @ 250 Hz)
  - Labels: `(n_samples,)`, values 0..4.

## Quick smoke test
Run the smoke test to verify environment and artifacts:
```
python smoke_test.py
```
It will:
- Load a few `.npz` files from `data/`.
- Initialize `RFInferenceEngine` and `CNNInferenceEngine` (if models exist).
- Run a tiny inference on 1 sample and report basic info.
