"""
Generate synthetic EEG dataset for mental command classification
"""
import numpy as np
import os
from config import SAMPLING_RATE, N_CHANNELS, WINDOW_LENGTH, CLASS_NAMES, DATA_DIR
def generate_synthetic_eeg_sample(class_label, n_channels=8, duration=2, fs=250, noise_std=0.15):
    """
    Generate a synthetic EEG sample for a specific mental command class
    Parameters:
    class_label : int
        Class identifier (0: neutral, 1: fan_on, 2: fan_off, 3: light_on, 4: light_off)
    n_channels : int
        Number of EEG channels
    duration : float
        Duration of signal in seconds
    fs : int
        Sampling frequency
    noise_std : float
        Standard deviation of noise
    Returns:
    signal : ndarray
        Generated EEG signal (n_channels, n_samples)
    """
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)
    # Initialize signal array
    signal = np.zeros((n_channels, n_samples))
    # Class-specific parameters
    if class_label == 0:  # neutral
        # Dominant in alpha range (8-12 Hz)
        dominant_freq = np.random.uniform(8, 12)
        secondary_freq = np.random.uniform(8, 13)
    elif class_label == 1:  # fan_on
        # Dominant in beta range (12-18 Hz)
        dominant_freq = np.random.uniform(12, 18)
        secondary_freq = np.random.uniform(13, 20)
    elif class_label == 2:  # fan_off
        # Dominant in high beta range (18-24 Hz)
        dominant_freq = np.random.uniform(18, 24)
        secondary_freq = np.random.uniform(15, 25)
    elif class_label == 3:  # light_on
        # Gamma component (24-28 Hz)
        dominant_freq = np.random.uniform(24, 28)
        secondary_freq = np.random.uniform(20, 30)
    elif class_label == 4:  # light_off
        # Mixed alpha-beta (10-15 Hz)
        dominant_freq = np.random.uniform(10, 15)
        secondary_freq = np.random.uniform(12, 18)
    # Generate signal for each channel
    for ch in range(n_channels):
        # Base signal with class-specific dominant frequency
        base_signal = np.sin(2 * np.pi * dominant_freq * t)
        # Add secondary frequency component
        secondary_amplitude = np.random.uniform(0.3, 0.7)
        base_signal += secondary_amplitude * np.sin(2 * np.pi * secondary_freq * t)
        # Add harmonics for realism
        harmonic_amplitude = np.random.uniform(0.1, 0.3)
        base_signal += harmonic_amplitude * np.sin(2 * np.pi * dominant_freq * 2 * t)
        # Add channel-specific variations
        phase_shift = np.random.uniform(0, 2 * np.pi)
        amplitude_mod = np.random.uniform(0.8, 1.2)
        signal[ch, :] = amplitude_mod * np.sin(2 * np.pi * dominant_freq * t + phase_shift)
        # Add secondary components with channel variations
        secondary_phase = np.random.uniform(0, 2 * np.pi)
        signal[ch, :] += secondary_amplitude * np.sin(2 * np.pi * secondary_freq * t + secondary_phase)
        # Add realistic noise
        noise = np.random.normal(0, noise_std, n_samples)
        signal[ch, :] += noise
        # Add slow drift to simulate baseline shifts
        drift = np.random.uniform(-0.1, 0.1) * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)
        signal[ch, :] += drift
    # Scale to realistic EEG amplitudes (-100 to +100 μV)
    signal = signal * np.random.uniform(15, 40)  # Adjusted scaling to keep within range
    return signal
def generate_synthetic_dataset(n_samples_per_class=650):
    """
    Generate complete synthetic EEG dataset
    Parameters:
    n_samples_per_class : int
        Number of samples to generate per class
    Returns:
    signals : ndarray
        EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    """
    n_timepoints = int(WINDOW_LENGTH * SAMPLING_RATE)  # 500 time points
    n_classes = len(CLASS_NAMES)
    total_samples = n_samples_per_class * n_classes
    # Pre-allocate arrays
    signals = np.zeros((total_samples, N_CHANNELS, n_timepoints))
    labels = np.zeros(total_samples, dtype=int)
    sample_idx = 0
    for class_label in range(n_classes):
        print(f"Generating {n_samples_per_class} samples for class {class_label} ({CLASS_NAMES[class_label]})")
        for i in range(n_samples_per_class):
            # Generate sample with class-specific characteristics
            signal = generate_synthetic_eeg_sample(class_label, N_CHANNELS, WINDOW_LENGTH, SAMPLING_RATE)
            signals[sample_idx] = signal
            labels[sample_idx] = class_label
            sample_idx += 1
    return signals, labels
def save_dataset(signals, labels, filepath):
    """
    Save dataset to NPZ file
    Parameters:
    signals : ndarray
        EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    filepath : str
        Path to save the dataset
    """
    np.savez_compressed(filepath, signals=signals, labels=labels)
    print(f"Dataset saved to {filepath}")
def validate_dataset(filepath):
    """
    Validate the generated dataset
    Parameters:
    filepath : str
        Path to the dataset file
    """
    print("\n=== DATASET VALIDATION ===")
    # Load dataset
    data = np.load(filepath)
    signals = data['signals']
    labels = data['labels']
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    # Check if shapes are correct
    n_timepoints = int(WINDOW_LENGTH * SAMPLING_RATE)
    expected_signals_shape = (signals.shape[0], N_CHANNELS, n_timepoints)
    expected_labels_shape = (signals.shape[0],)
    if signals.shape == expected_signals_shape:
        print("✓ Signals shape is correct")
    else:
        print(f"✗ Signals shape mismatch. Expected: {expected_signals_shape}, Got: {signals.shape}")
    if labels.shape == expected_labels_shape:
        print("✓ Labels shape is correct")
    else:
        print(f"✗ Labels shape mismatch. Expected: {expected_labels_shape}, Got: {labels.shape}")
    # Check sample count per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nSamples per class:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label} ({CLASS_NAMES[label]}): {count} samples")
    # Check if we have sufficient samples per class
    min_samples_per_class = 600
    std_dev = np.std(counts)
    print(f"\nStandard deviation of class counts: {std_dev:.2f}")
    if all(count >= min_samples_per_class for count in counts) and std_dev < 50:
        print("✓ Sufficient samples per class (>= 600) with low standard deviation (< 50)")
    else:
        print("✗ Sample distribution does not meet requirements")
    # Check signal amplitudes
    min_amp = np.min(signals)
    max_amp = np.max(signals)
    print(f"\nSignal amplitude range: [{min_amp:.2f}, {max_amp:.2f}] μV")
    if -100 <= min_amp and max_amp <= 100:
        print("✓ Signal amplitudes are within realistic range (-100 to +100 μV)")
    else:
        print("✗ Signal amplitudes are outside realistic range")
    # Check if all class labels are present
    expected_labels = set(range(len(CLASS_NAMES)))
    actual_labels = set(unique_labels)
    if expected_labels == actual_labels:
        print("✓ All expected class labels (0-4) are present")
    else:
        print(f"✗ Missing class labels. Expected: {expected_labels}, Got: {actual_labels}")
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total samples: {signals.shape[0]}")
    print(f"Number of channels: {signals.shape[1]}")
    print(f"Time points per sample: {signals.shape[2]}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Window length: {WINDOW_LENGTH} seconds")
if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic EEG dataset...")
    signals, labels = generate_synthetic_dataset(n_samples_per_class=650)
    # Save dataset
    dataset_filepath = os.path.join(DATA_DIR, "synthetic_eeg_data.npz")
    save_dataset(signals, labels, dataset_filepath)
    # Validate dataset
    validate_dataset(dataset_filepath)
    print("\nDataset generation and validation completed!")
