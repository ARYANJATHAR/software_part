"""
Real-time inference script for CNN EEG classifier
"""
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
# Add project directory to path
sys.path.append('/app/eeg_command_classifier_0728')
# Import our modules
from config import *
from preprocessing import bandpass_filter, normalize_signal
class EEG1DCNN(nn.Module):
    """
    1D CNN model for EEG classification (matching the training model)
    """
    def __init__(self, n_channels=8, n_timepoints=500, n_classes=5):
        super(EEG1DCNN, self).__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Calculate the size after convolutions and pooling
        # Initial size: 500
        # After pool1: 250
        # After pool2: 125
        self.flatten_size = 64 * 125  # 64 channels * 125 timepoints
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
    def forward(self, x):
        # x shape: (batch_size, n_channels, n_timepoints)
        x = self.conv1(x)  # (batch_size, 32, 500)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 32, 250)
        x = self.conv2(x)  # (batch_size, 64, 250)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 64, 125)
        x = x.view(-1, self.flatten_size)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class CNNInferenceEngine:
    """Real-time inference engine for CNN EEG classifier"""
    def __init__(self, model_path=None, device=None):
        """
        Initialize the inference engine
        Parameters:
        model_path (str): Path to the trained model file
        device (torch.device): Device to run inference on
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        # Initialize model
        self.model = EEG1DCNN(n_channels=N_CHANNELS, n_timepoints=500, n_classes=N_CLASSES)
        # Load model weights
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'cnn_model.pth')
        # Load model state dict
        model_state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state)
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        # Store configuration
        self.sampling_rate = SAMPLING_RATE
        self.n_channels = N_CHANNELS
        self.n_classes = N_CLASSES
        self.bandpass_range = BANDPASS_RANGE
        self.window_length = WINDOW_LENGTH
        self.class_labels = CLASS_LABELS
        self.class_names = CLASS_NAMES
        print(f"CNN model loaded successfully")
        print(f"Model classes: {self.class_names}")
    def preprocess_signal(self, signal_data):
        """
        Apply preprocessing pipeline to EEG signal
        Parameters:
        signal_data (np.ndarray): Raw EEG signal of shape (n_channels, n_samples)
        Returns:
        torch.Tensor: Preprocessed signal tensor
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
        # Convert to tensor and add batch dimension
        signal_tensor = torch.FloatTensor(normalized_signal).unsqueeze(0)  # Add batch dimension
        signal_tensor = signal_tensor.to(self.device)
        return signal_tensor
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
        signal_tensor = self.preprocess_signal(signal_data)
        # Make prediction
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        # Convert to CPU for processing
        prediction = prediction.cpu().item()
        probabilities = probabilities.cpu().numpy().flatten()
        if return_proba:
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
        # Convert to tensor
        batch_tensor = torch.FloatTensor(signal_batch).to(self.device)
        # Make predictions
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        # Convert to CPU for processing
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        # Convert to class labels
        pred_labels = [self.class_labels[pred] for pred in predictions]
        if return_proba:
            return pred_labels, probabilities
        else:
            return pred_labels
    def warmup(self, n_warmup=3):
        """
        Warm up the model with dummy inferences to avoid first-run overhead
        Parameters:
        n_warmup (int): Number of warmup inferences
        """
        print("Warming up model...")
        dummy_signal = np.random.randn(1, self.n_channels, 500).astype(np.float32)
        dummy_tensor = torch.FloatTensor(dummy_signal).to(self.device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_tensor)
        print("Model warmup completed!")
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
    print("Initializing CNN EEG Inference Engine...")
    # Initialize inference engine
    engine = CNNInferenceEngine()
    # Warm up model
    engine.warmup()
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
    # Test batch processing
    print("\nTesting batch processing (batch_size=5):")
    batch_size = min(5, len(signals))
    batch_signals = signals[:batch_size]
    batch_labels = labels[:batch_size]
    start_time = time.time()
    pred_labels, all_probabilities = engine.process_batch(batch_signals, return_proba=True)
    end_time = time.time()
    batch_latency = (end_time - start_time) * 1000  # in milliseconds
    print(f"Batch processing latency: {batch_latency:.2f} ms for {batch_size} samples")
    print(f"Average per-sample latency in batch: {batch_latency / batch_size:.2f} ms")
    # Test with simulated data
    print("\nTesting with simulated EEG signal:")
    simulated_signal = simulate_eeg_signal()
    pred_label, probabilities = engine.predict(simulated_signal, return_proba=True)
    confidence = np.max(probabilities)
    print(f"Predicted: {pred_label} (Confidence: {confidence:.3f})")
    print("\nCNN inference test completed successfully!")
if __name__ == "__main__":
    main()
