"""
Feature extraction module for EEG signal processing
"""
import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft, fftfreq
def extract_time_domain_features(signal_data):
    """
    Extract time-domain features from EEG signal
    Parameters:
    signal_data : array_like
        Input signal (n_channels, n_samples) or (n_samples,)
    Returns:
    features : ndarray
        Time-domain features
    """
    # Handle both 1D and 2D inputs
    if len(signal_data.shape) == 1:
        signal_data = signal_data.reshape(1, -1)
    n_channels, n_samples = signal_data.shape
    features = []
    for ch in range(n_channels):
        ch_data = signal_data[ch, :]
        # Basic statistical features
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        var_val = np.var(ch_data)
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        # Energy and RMS
        energy = np.sum(ch_data ** 2)
        rms = np.sqrt(np.mean(ch_data ** 2))
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
        # Skewness and kurtosis
        skewness = stats.skew(ch_data)
        kurt = stats.kurtosis(ch_data)
        channel_features = [mean_val, std_val, var_val, min_val, max_val,
                            energy, rms, zero_crossings, skewness, kurt]
        features.extend(channel_features)
    return np.array(features)
def extract_frequency_domain_features(signal_data, sampling_rate, fft_size=512):
    """
    Extract frequency-domain features from EEG signal
    Parameters:
    signal_data : array_like
        Input signal (n_channels, n_samples) or (n_samples,)
    sampling_rate : int
        Sampling rate in Hz
    fft_size : int
        Size of FFT
    Returns:
    features : ndarray
        Frequency-domain features
    """
    # Handle both 1D and 2D inputs
    if len(signal_data.shape) == 1:
        signal_data = signal_data.reshape(1, -1)
    n_channels, n_samples = signal_data.shape
    features = []
    # Frequency bands
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
    gamma_band = (30, 45)
    for ch in range(n_channels):
        ch_data = signal_data[ch, :]
        # Pad or truncate signal to fft_size
        if len(ch_data) < fft_size:
            ch_data = np.pad(ch_data, (0, fft_size - len(ch_data)))
        else:
            ch_data = ch_data[:fft_size]
        # Compute FFT
        fft_vals = fft(ch_data)
        fft_freq = fftfreq(fft_size, 1 / sampling_rate)
        # Only use positive frequencies
        positive_freq_idx = fft_freq >= 0
        fft_vals = fft_vals[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]
        # Power spectral density
        psd = np.abs(fft_vals) ** 2
        # Compute band powers
        def band_power(freq, psd, band):
            band_idx = (freq >= band[0]) & (freq < band[1])
            return np.sum(psd[band_idx])
        delta_power = band_power(fft_freq, psd, delta_band)
        theta_power = band_power(fft_freq, psd, theta_band)
        alpha_power = band_power(fft_freq, psd, alpha_band)
        beta_power = band_power(fft_freq, psd, beta_band)
        gamma_power = band_power(fft_freq, psd, gamma_band)
        # Total power
        total_power = np.sum(psd)
        # Peak frequency in beta band (8-30 Hz)
        beta_idx = (fft_freq >= 8) & (fft_freq <= 30)
        if np.any(beta_idx) and np.sum(psd[beta_idx]) > 0:
            peak_freq_idx = np.argmax(psd[beta_idx])
            peak_frequency = fft_freq[beta_idx][peak_freq_idx]
        else:
            peak_frequency = 0
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zero values for log
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
        channel_features = [delta_power, theta_power, alpha_power, beta_power,
                            gamma_power, total_power, peak_frequency, spectral_entropy]
        features.extend(channel_features)
    return np.array(features)
def extract_all_features(signal_data, sampling_rate, fft_size=512):
    """
    Extract all features (time and frequency domain) from EEG signal
    Parameters:
    signal_data : array_like
        Input signal (n_channels, n_samples)
    sampling_rate : int
        Sampling rate in Hz
    fft_size : int
        Size of FFT
    Returns:
    features : ndarray
        Combined time and frequency features
    """
    time_features = extract_time_domain_features(signal_data)
    freq_features = extract_frequency_domain_features(signal_data, sampling_rate, fft_size)
    # Combine all features
    all_features = np.concatenate([time_features, freq_features])
    return all_features
# Test function to verify the module works
if __name__ == "__main__":
    # Generate test signal
    t = np.linspace(0, 2, 500)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    test_signal = test_signal.reshape(1, -1)  # Reshape to (n_channels, n_samples)
    # Extract features
    time_features = extract_time_domain_features(test_signal)
    freq_features = extract_frequency_domain_features(test_signal, 250)
    all_features = extract_all_features(test_signal, 250)
    print("Feature extraction module functions working correctly")
    print(f"Time-domain features shape: {time_features.shape}")
    print(f"Frequency-domain features shape: {freq_features.shape}")
    print(f"All features shape: {all_features.shape}")
    print(f"Sample time features: {time_features[:5]}")
    print(f"Sample frequency features: {freq_features[:5]}")
