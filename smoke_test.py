"""
Quick smoke test for EEG Command Classifier workspace
- Verifies data files exist and can be loaded
- Initializes RF and CNN inference engines (if models exist)
- Runs a tiny inference on one validation sample
"""
import os
import numpy as np
from config import DATA_DIR, MODEL_DIR, CLASS_LABELS, CLASS_NAMES

errors = []

print("== Smoke Test ==")
print(f"DATA_DIR:   {DATA_DIR}")
print(f"MODEL_DIR:  {MODEL_DIR}")

# Check data files
data_files = [
    os.path.join(DATA_DIR, "signals_val.npz"),
    os.path.join(DATA_DIR, "features_val.npz"),
]
for f in data_files:
    if not os.path.exists(f):
        errors.append(f"Missing data file: {f}")
    else:
        try:
            d = np.load(f)
            print(f"Loaded {os.path.basename(f)}: keys={list(d.keys())}")
        except Exception as e:
            errors.append(f"Failed to load {f}: {e}")

# Try to init RF engine
rf_model_path = os.path.join(MODEL_DIR, "random_forest_baseline.pkl")
rf_ok = False
try:
    if os.path.exists(rf_model_path):
        from inference import RFInferenceEngine
        rf_engine = RFInferenceEngine(model_path=rf_model_path)
        rf_ok = True
        print("RFInferenceEngine init: OK")
    else:
        print("RF model not found; skipping RF init.")
except Exception as e:
    errors.append(f"RFInferenceEngine init failed: {e}")

# Try to init CNN engine
cnn_model_path = os.path.join(MODEL_DIR, "cnn_model.pth")
cnn_ok = False
try:
    if os.path.exists(cnn_model_path):
        from inference_cnn import CNNInferenceEngine
        cnn_engine = CNNInferenceEngine(model_path=cnn_model_path)
        cnn_ok = True
        print("CNNInferenceEngine init: OK")
    else:
        print("CNN model not found; skipping CNN init.")
except Exception as e:
    errors.append(f"CNNInferenceEngine init failed: {e}")

# Try tiny inference using first val sample
sig_val_path = os.path.join(DATA_DIR, "signals_val.npz")
if os.path.exists(sig_val_path):
    try:
        val = np.load(sig_val_path)
        signals = val["signals"]
        labels = val["labels"]
        if len(signals) > 0:
            sample = signals[0]
            true_label = labels[0]
            print(f"First val sample shape: {sample.shape}; true={CLASS_LABELS[int(true_label)]}")
            if rf_ok:
                pred_label, proba = rf_engine.predict(sample, return_proba=True)
                print(f"RF predict: {pred_label} (conf={np.max(proba):.2f})")
            if cnn_ok:
                pred_label, proba = cnn_engine.predict(sample, return_proba=True)
                print(f"CNN predict: {pred_label} (conf={np.max(proba):.2f})")
    except Exception as e:
        errors.append(f"Tiny inference failed: {e}")
else:
    print("signals_val.npz missing; skipping tiny inference.")

if errors:
    print("\nSmoke test completed with issues:")
    for e in errors:
        print(" -", e)
else:
    print("\nSmoke test passed: basic checks OK.")
