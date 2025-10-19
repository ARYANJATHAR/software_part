"""
Configuration file for EEG Command Classifier Pipeline
"""
# EEG Signal Parameters
SAMPLING_RATE = 250  # Hz
N_CHANNELS = 8       # Number of EEG channels
N_CLASSES = 5        # Number of mental command classes
BANDPASS_RANGE = (8, 30)  # Hz - Alpha and Beta bands
WINDOW_LENGTH = 2    # seconds
# Class labels for mental commands
CLASS_LABELS = {
    0: 'neutral',
    1: 'fan_on',
    2: 'fan_off',
    3: 'light_on',
    4: 'light_off'
}
# Class names for plotting and reporting
CLASS_NAMES = ['neutral', 'fan_on', 'fan_off', 'light_on', 'light_off']
# Data directory paths (repo-relative for portability)
import os
_BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(_BASE_DIR, 'data')
MODEL_DIR = os.path.join(_BASE_DIR, 'models')
RESULTS_DIR = os.path.join(_BASE_DIR, 'results')
# Model parameters
RF_N_ESTIMATORS = 100
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LR = 0.001
# Feature extraction parameters
FFT_SIZE = 512
