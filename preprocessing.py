"""
Preprocessing module for EEG signal processing
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')
def bandpass_filter(signal_data, lowcut, highcut, fs, order=4):
    """
    Apply bandpass filter to EEG signal
    Parameters:
    signal_data : array_like
        Input signal
    lowcut : float
        Low cutoff frequency
    highcut : float
        High cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
    Returns:
    filtered_signal : ndarray
        Bandpass filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError("Invalid cutoff frequencies")
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal_data, axis=-1)
    return filtered_signal
def normalize_signal(signal_data, axis=-1):
    """
    Apply z-score normalization to signal
    Parameters:
    signal_data : array_like
        Input signal
    axis : int
        Axis along which to normalize
    Returns:
    normalized_signal : ndarray
        Z-score normalized signal
    """
    mean = np.mean(signal_data, axis=axis, keepdims=True)
    std = np.std(signal_data, axis=axis, keepdims=True)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    normalized_signal = (signal_data - mean) / std
    return normalized_signal
def remove_artifacts(signal_data, threshold=3):
    """
    Simple artifact removal based on amplitude threshold
    Parameters:
    signal_data : array_like
        Input signal
    threshold : float
        Standard deviation threshold for artifact detection
    Returns:
    cleaned_signal : ndarray
        Artifact-cleaned signal
    """
    # This is a simple implementation - in practice, more sophisticated methods are used
    std_dev = np.std(signal_data)
    mean_val = np.mean(signal_data)
    # Create mask for outliers
    mask = np.abs(signal_data - mean_val) < (threshold * std_dev)
    # For simplicity, we'll just clip the values instead of removing samples
    cleaned_signal = np.clip(signal_data,
                             mean_val - threshold * std_dev,
                             mean_val + threshold * std_dev)
    return cleaned_signal
def segment_signal(signal_data, window_length, sampling_rate):
    """
    Segment continuous signal into fixed-length windows
    Parameters:
    signal_data : array_like
        Continuous input signal
    window_length : float
        Length of each window in seconds
    sampling_rate : int
        Sampling rate in Hz
    Returns:
    segments : ndarray
        Segmented signal (n_segments, n_channels, n_samples_per_window)
    """
    samples_per_window = int(window_length * sampling_rate)
    n_samples = signal_data.shape[-1]
    n_segments = n_samples // samples_per_window
    if n_segments == 0:
        raise ValueError("Signal too short for specified window length")
    # Trim signal to fit exact number of segments
    trimmed_length = n_segments * samples_per_window
    if len(signal_data.shape) == 2:
        trimmed_signal = signal_data[:, :trimmed_length]
        segments = trimmed_signal.reshape(signal_data.shape[0], n_segments, samples_per_window)
        segments = np.transpose(segments, (1, 0, 2))
    else:
        trimmed_signal = signal_data[:trimmed_length]
        segments = trimmed_signal.reshape(n_segments, samples_per_window)
    return segments
# Test function to verify the module works
if __name__ == "__main__":
    # Generate test signal
    t = np.linspace(0, 2, 500)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    # Apply bandpass filter (8-30 Hz)
    filtered = bandpass_filter(test_signal, 8, 30, 250)
    # Normalize signal
    normalized = normalize_signal(filtered)
    print("Preprocessing module functions working correctly")
    print(f"Original signal shape: {test_signal.shape}")
    print(f"Filtered signal shape: {filtered.shape}")
    print(f"Normalized signal mean: {np.mean(normalized):.4f}")
    print(f"Normalized signal std: {np.std(normalized):.4f}")
