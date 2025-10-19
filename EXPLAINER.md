# Brain‑Controlled Smart Glasses – Project Explainer

A practical, end‑to‑end machine learning pipeline that reads brainwave‑like EEG signals and turns them into everyday smart‑home commands such as “fan on”, “fan off”, “light on”, “light off”, or “neutral”. The system is designed to be lightweight, fast, and ready to plug into a mobile or web app.

---

## What this project is
This repository builds a complete software stack for recognizing mental commands from EEG data. It focuses on the data and ML side (preprocessing, features, models, and inference). Hardware can be real EEG sensors or simulated data for development.

- Input: multichannel EEG segments (8 channels, 2 seconds at 250 Hz → 500 samples per segment)
- Output: one of five commands: neutral, fan_on, fan_off, light_on, light_off
- Two model paths:
  - A classic baseline using Random Forest on hand‑crafted features
  - A lightweight 1D Convolutional Neural Network (CNN) on raw signals
- Real‑time friendly: low‑latency inference and batch processing supported

Why it matters: It opens the door to hands‑free interaction for accessibility, AR/VR, and smart‑home control by translating thought patterns into actions.

---

## What it can do
- Generate realistic synthetic EEG‑like data for 5 classes (useful when hardware data isn’t available)
- Clean and normalize EEG signals with standard steps: band‑pass 8–30 Hz, artifact clipping, z‑score normalization
- Extract meaningful features (time and frequency domain) for classical models
- Train and evaluate two complementary classifiers: Random Forest and 1D CNN
- Export trained models (RF: `.pkl`, CNN: `.pth`) and metrics/plots
- Run inference on validation data and on simulated live signals
- Visualize signals, class probabilities, frequency spectrum, and confusion matrix

---

## Why it’s helpful
- Rapid prototyping: Start with synthetic data, then swap in real EEG later without changing the pipeline
- Low‑latency: CNN inference achieves sub‑millisecond to a few milliseconds per sample on CPU in batch mode
- Clear artifacts: All steps produce saved files (datasets, models, metrics, plots) for inspection and reproducibility
- Deployable: Clean separation of preprocessing and inference. Easy to wrap into a Streamlit demo or a mobile app that receives EEG via Bluetooth Low Energy (BLE)

---

## How it works (the pipeline)
1) Data generation (optional) – `generate_synthetic_eeg.py`
   - Creates EEG‑like signals per class with distinct dominant frequency bands, noise, and drift
   - Validates shape, class balance, and amplitude ranges

2) Preprocessing – `preprocess_data.py` (uses `preprocessing.py`)
   - Band‑pass filter (8–30 Hz) to capture alpha/beta rhythms
   - Simple artifact clipping by amplitude threshold
   - Z‑score normalization per channel
   - Validation checks confirm mean ≈ 0, std ≈ 1, and correct frequency content

3) Feature extraction (for classical models) – `extract_features.py` (uses `feature_extraction.py`)
   - Time domain: mean, variance, rms, energy, zero‑crossing rate, skewness, kurtosis
   - Frequency domain: delta–gamma band powers, total power, spectral entropy, peak beta frequency

4) Split train/val – `split_data.py`
   - Stratified split using shared indices for both feature and signal datasets

5) Training
   - Random Forest: `train_random_forest.py` trains on features, exports `.pkl` and metrics
   - 1D CNN: `train_cnn.py` trains on signals, early stopping, exports `.pth` and training history/plots

6) Inference & demo
   - RF inference: `inference.py` – preprocess → features → predict(+proba)
   - CNN inference: `inference_cnn.py` – preprocess → predict(+proba); supports batch and warmup
   - Demo: `demo_inference.py` – side‑by‑side comparison with timings and visual outputs (see `results/`)
   - Visualization: `visualize_predictions.py` – signals, probabilities, spectrum, history, confusion matrix

All paths are repo‑relative (works on Windows/macOS/Linux) via `config.py`.

---

## Performance snapshot (from a typical synthetic run)
- Random Forest (features, CPU):
  - Validation accuracy ≈ 0.89 (macro F1 ≈ 0.89)
  - Inference latency ≈ 30–40 ms per sample
- 1D CNN (raw signals, CPU):
  - Validation accuracy ≈ 0.81 (macro F1 ≈ 0.81)
  - Inference latency ≈ 1.5–2.0 ms per sample (single), ≈ 0.5 ms per sample in batch

Notes: Accuracy depends on data realism and class definitions. Latency improves further with GPU or model export (TorchScript/ONNX) and quantization.

---

## How to use it
1) Install dependencies
```
pip install -r requirements.txt
```
2) Run the end‑to‑end synthetic pipeline
```
python generate_synthetic_eeg.py
python preprocess_data.py
python extract_features.py
python split_data.py
python train_random_forest.py
python train_cnn.py
```
3) Try inference and demos
```
python inference.py        # Random Forest
python inference_cnn.py    # CNN
python demo_inference.py   # Side‑by‑side comparison + visuals
```
Artifacts are saved under `models/` and `results/`.

---

## Integrating with apps (Streamlit/mobile)
- Wrap the inference engine in an app loop that receives 2‑second EEG windows over BLE/WebSocket
- Preprocess with the same steps (band‑pass 8–30 Hz, z‑score); then:
  - RF path: extract features → predict → trigger action
  - CNN path: reshape to (channels, time) → predict → trigger action
- Use class probabilities for UI feedback (confidence bars, smoothing over time)

---

## Limitations and considerations
- Synthetic data ≠ real EEG: Real‑world noise/artifacts (eye blinks, muscle) and placement variability require more robust preprocessing (ICA, artifact subspace, adaptive filters)
- Calibration may be needed per user: A short per‑user fine‑tuning or normalization pass can improve accuracy
- Safety: Always keep manual overrides and confirmations for critical IoT actions

---

## What to improve next
- Add a continuous streaming inference loop with sliding windows (e.g., 250 ms hop) for a live demo
- Export CNN to TorchScript/ONNX and try int8 quantization for mobile speed‑ups
- Add more features (e.g., wavelet coefficients) and try other models (SVM, LightGBM)
- Add simple unit tests and a reproducible seed setup
- Integrate an actual BLE data source or a mock streaming source for the UI

---

## FAQ
- Can I use a public dataset instead of synthetic data?
  Yes. Replace the synthetic `.npz` with arrays shaped like `(n_samples, 8, 500)` for signals and `(n_samples,)` for labels. Then run from preprocessing onward.

- Do I need a GPU?
  No. The CNN runs fast on CPU for this window size. A GPU helps for training larger models or datasets.

- How hard is it to connect this to a Streamlit app?
  Easy. Import the inference engine, collect a 2‑second window from BLE or a buffer, call `predict`, and update the UI.
