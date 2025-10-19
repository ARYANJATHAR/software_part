"""
Real-time inference script for Random Forest EEG classifier
"""
import numpy as np
import joblib
import time
from scipy import signal
import sys
import os
# Add project directory to path
sys.path.append('/app/eeg_command_classifier_0728')
# Import our modules
from config import *
from preprocessing import bandpass_filter, normalize_signal
from feature_extraction import extract_all_features
class RFInferenceEngine:
    """Real-time inference engine for Random Forest EEG classifier"""
    def __init__(self, model_path=None):
        """
        Initialize the inference engine
        Parameters:
        model_path (str): Path to the trained model file
        """
        # Load model
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'random_forest_baseline.pkl')
        self.model = joblib.load(model_path)
        # Store configuration
        self.sampling_rate = SAMPLING_RATE
        self.n_channels = N_CHANNELS
        self.n_classes = N_CLASSES
        self.bandpass_range = BANDPASS_RANGE
        self.window_length = WINDOW_LENGTH
        self.class_labels = CLASS_LABELS
        self.class_names = CLASS_NAMES
        print(f"Random Forest model loaded successfully")
        print(f"Model classes: {self.class_names}")
    def preprocess_signal(self, signal_data):
        """
        Apply preprocessing pipeline to EEG signal
        Parameters:
        signal_data (np.ndarray): Raw EEG signal of shape (n_channels, n_samples)
        Returns:
        np.ndarray: Preprocessed signal
        """
        # Apply bandpass filter (8-30 Hz)
        filtered_signal = bandpass_filter(
            signal_data,
            self.bandpass_range[0],
            self.bandpass_range[1],
            self.sampling_rate
        )
        # Normalize signal (z-score)
        normalized_signal = normalize_signal(filtered_signal)
        return normalized_signal
    def extract_features(self, signal_data):
        """
        Extract features from preprocessed EEG signal
        Parameters:
        signal_data (np.ndarray): Preprocessed EEG signal of shape (n_channels, n_samples)
        Returns:
        np.ndarray: Feature vector
        """
        # Extract all features
        features = extract_all_features(
            signal_data,
            self.sampling_rate,
            FFT_SIZE
        )
        return features
    def predict(self, signal_data, return_proba=False):
        """
        Make prediction on EEG signal
        Parameters:
        signal_data (np.ndarray): Raw EEG signal of shape (n_channels, n_samples)
        return_proba (bool): Whether to return prediction probabilities
        Returns:
        str or tuple: Predicted class label (and probabilities if return_proba=True)
        """
        # Preprocess signal
        preprocessed_signal = self.preprocess_signal(signal_data)
        # Extract features
        features = self.extract_features(preprocessed_signal)
        # Make prediction
        prediction = self.model.predict([features])[0]
        if return_proba:
            # Get prediction probabilities
            probabilities = self.model.predict_proba([features])[0]
            return self.class_labels[prediction], probabilities
        else:
            return self.class_labels[prediction]
    def process_batch(self, signal_batch, return_proba=False):
        """
        Process a batch of EEG signals
        Parameters:
        signal_batch (np.ndarray): Batch of raw EEG signals of shape (batch_size, n_channels, n_samples)
        return_proba (bool): Whether to return prediction probabilities
        Returns:
        list or tuple: List of predicted class labels (and probabilities if return_proba=True)
        """
        predictions = []
        if return_proba:
            all_probabilities = []
        for i in range(signal_batch.shape[0]):
            signal_data = signal_batch[i]
            result = self.predict(signal_data, return_proba)
            if return_proba:
                pred_label, probabilities = result
                predictions.append(pred_label)
                all_probabilities.append(probabilities)
            else:
                predictions.append(result)
        if return_proba:
            return predictions, all_probabilities
        else:
            return predictions
def simulate_eeg_signal(duration=2, sampling_rate=250, n_channels=8):
    """
    Generate simulated EEG signal for testing
    Parameters:
    duration (float): Signal duration in seconds
    sampling_rate (int): Sampling rate in Hz
    n_channels (int): Number of EEG channels
    Returns:
    np.ndarray: Simulated EEG signal of shape (n_channels, n_samples)
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    # Create multi-channel signal with different patterns for different classes
    signal_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Base signal with multiple frequency components
        signal_data[ch] = (
            np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
            0.5 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz component
            0.3 * np.sin(2 * np.pi * 30 * t) +  # 30 Hz component
            0.1 * np.random.randn(n_samples)  # Noise
        )
    return signal_data
def main():
    """Main function to demonstrate inference"""
    print("Initializing Random Forest EEG Inference Engine...")
    # Initialize inference engine
    engine = RFInferenceEngine()
    # Load validation data for testing
    print("Loading validation data...")
    val_data = np.load(os.path.join(DATA_DIR, 'signals_val.npz'))
    signals = val_data['signals']
    labels = val_data['labels']
    print(f"Validation data shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    # Test on a few samples
    print("\nTesting inference on validation samples:")
    correct = 0
    total = min(10, len(signals))  # Test on first 10 samples
    start_time = time.time()
    for i in range(total):
        # Get a sample
        signal_sample = signals[i]  # Shape: (8, 500)
        true_label = labels[i]
        # Make prediction
        pred_label, probabilities = engine.predict(signal_sample, return_proba=True)
        confidence = np.max(probabilities)
        # Check if correct
        if CLASS_NAMES.index(pred_label) == true_label:
            correct += 1
        print(f"Sample {i + 1}: True={CLASS_LABELS[true_label]}, "
              f"Predicted={pred_label} (Confidence: {confidence:.3f})")
    end_time = time.time()
    avg_latency = (end_time - start_time) / total * 1000  # in milliseconds
    accuracy = correct / total * 100
    print(f"\nAccuracy on {total} samples: {accuracy:.1f}%")
    print(f"Average inference latency: {avg_latency:.2f} ms")
    # Test with simulated data
    print("\nTesting with simulated EEG signal:")
    simulated_signal = simulate_eeg_signal()
    pred_label, probabilities = engine.predict(simulated_signal, return_proba=True)
    confidence = np.max(probabilities)
    print(f"Predicted: {pred_label} (Confidence: {confidence:.3f})")
    print("\nInference test completed successfully!")
if __name__ == "__main__":
    main()
