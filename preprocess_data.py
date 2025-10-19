"""
Preprocess synthetic EEG dataset for mental command classification
"""
import numpy as np
import os
from scipy import signal
from config import SAMPLING_RATE, BANDPASS_RANGE, DATA_DIR
import preprocessing  # Our custom preprocessing module
def preprocess_eeg_dataset(signals, labels):
    """
    Apply preprocessing pipeline to EEG dataset
    Parameters:
    signals : ndarray
        EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    Returns:
    processed_signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    processed_labels : ndarray
        Class labels (n_samples,)
    """
    n_samples, n_channels, n_timepoints = signals.shape
    processed_signals = np.zeros_like(signals)
    print("Applying preprocessing pipeline...")
    # Process each sample
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"  Processing sample {i}/{n_samples}")
        # Get the current sample
        sample = signals[i]  # shape: (n_channels, n_timepoints)
        # 1. Apply bandpass filter (8-30 Hz)
        filtered_sample = preprocessing.bandpass_filter(
            sample, BANDPASS_RANGE[0], BANDPASS_RANGE[1], SAMPLING_RATE
        )
        # 2. Apply amplitude-based artifact removal
        cleaned_sample = preprocessing.remove_artifacts(filtered_sample, threshold=3)
        # 3. Normalize each channel using z-score
        normalized_sample = preprocessing.normalize_signal(cleaned_sample, axis=1)
        # Store processed sample
        processed_signals[i] = normalized_sample
    print("Preprocessing completed!")
    return processed_signals, labels
def validate_preprocessed_data(original_signals, processed_signals, labels):
    """
    Validate preprocessed EEG data
    Parameters:
    original_signals : ndarray
        Original EEG signals (n_samples, n_channels, n_timepoints)
    processed_signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    """
    print("\n=== PREPROCESSING VALIDATION ===")
    # Check shapes
    print(f"Original signals shape: {original_signals.shape}")
    print(f"Processed signals shape: {processed_signals.shape}")
    print(f"Labels shape: {labels.shape}")
    if original_signals.shape == processed_signals.shape:
        print("✓ Signal shapes match")
    else:
        print("✗ Signal shapes don't match")
    # Check normalization - mean ≈ 0 and std ≈ 1 for each channel
    n_samples, n_channels, n_timepoints = processed_signals.shape
    sample_indices = [0, 100, 500, 1000, 2000]  # Check a few samples
    print("\nSample normalization statistics:")
    for idx in sample_indices:
        sample = processed_signals[idx]
        channel_means = np.mean(sample, axis=1)
        channel_stds = np.std(sample, axis=1)
        print(f"  Sample {idx} (Class {labels[idx]}):")
        print(f"    Channel means: {channel_means}")
        print(f"    Channel stds: {channel_stds}")
    # Compute overall statistics
    all_channel_means = np.mean(processed_signals, axis=(0, 2))  # Mean across samples and time
    all_channel_stds = np.std(processed_signals, axis=(0, 2))    # Std across samples and time
    print(f"\nOverall channel means: {all_channel_means}")
    print(f"Overall channel stds: {all_channel_stds}")
    # Check if normalized (mean ≈ 0, std ≈ 1)
    mean_check = np.allclose(all_channel_means, 0, atol=0.1)
    std_check = np.allclose(all_channel_stds, 1, atol=0.1)
    if mean_check:
        print("✓ Channel means are approximately 0")
    else:
        print("✗ Channel means are not close to 0")
    if std_check:
        print("✓ Channel standard deviations are approximately 1")
    else:
        print("✗ Channel standard deviations are not close to 1")
    # Frequency content validation using FFT
    print("\nFrequency content validation (sampling 10 samples):")
    valid_freq_content = True
    for i in range(0, n_samples, n_samples // 10):  # Sample 10 signals
        sample = processed_signals[i]
        # Take FFT of first channel
        fft_vals = np.fft.fft(sample[0])
        fft_freq = np.fft.fftfreq(n_timepoints, 1 / SAMPLING_RATE)
        # Only positive frequencies
        positive_freq_idx = fft_freq >= 0
        fft_vals = fft_vals[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]
        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_vals))
        peak_freq = fft_freq[peak_idx]
        # Check if peak frequency is within bandpass range
        if not (BANDPASS_RANGE[0] <= peak_freq <= BANDPASS_RANGE[1]):
            valid_freq_content = False
            print(f"  Sample {i}: Peak frequency {peak_freq:.2f} Hz outside bandpass range")
    if valid_freq_content:
        print("✓ Frequency content is mostly within 8-30 Hz bandpass range")
    else:
        print("✗ Some samples have frequency content outside 8-30 Hz range")
    # Verify segmentation (should be 2 seconds each)
    expected_timepoints = int(2 * SAMPLING_RATE)  # 500 time points for 2 seconds
    if n_timepoints == expected_timepoints:
        print(f"✓ Segmentation verified: {n_timepoints} timepoints ({2} seconds)")
    else:
        print(f"✗ Segmentation issue: {n_timepoints} timepoints (expected {expected_timepoints})")
def save_preprocessed_data(signals, labels, filepath):
    """
    Save preprocessed data to NPZ file
    Parameters:
    signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    filepath : str
        Path to save the dataset
    """
    np.savez_compressed(filepath, signals=signals, labels=labels)
    print(f"\nPreprocessed data saved to {filepath}")
def load_dataset(filepath):
    """
    Load dataset from NPZ file
    Parameters:
    filepath : str
        Path to the dataset file
    Returns:
    signals : ndarray
        EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    """
    data = np.load(filepath)
    signals = data['signals']
    labels = data['labels']
    print(f"Dataset loaded from {filepath}")
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    return signals, labels
if __name__ == "__main__":
    # Load synthetic EEG dataset
    input_filepath = os.path.join(DATA_DIR, "synthetic_eeg_data.npz")
    signals, labels = load_dataset(input_filepath)
    # Apply preprocessing pipeline
    processed_signals, processed_labels = preprocess_eeg_dataset(signals, labels)
    # Validate preprocessed data
    validate_preprocessed_data(signals, processed_signals, labels)
    # Save preprocessed data
    output_filepath = os.path.join(DATA_DIR, "preprocessed_eeg_data.npz")
    save_preprocessed_data(processed_signals, processed_labels, output_filepath)
    print("\nPreprocessing pipeline completed successfully!")
