"""
Extract time and frequency domain features from preprocessed EEG data
"""
import numpy as np
import os
from config import SAMPLING_RATE, DATA_DIR
import feature_extraction  # Our custom feature extraction module
def extract_features_from_dataset(signals, labels):
    """
    Extract features from EEG dataset
    Parameters:
    signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    Returns:
    features : ndarray
        Extracted features (n_samples, n_features)
    feature_labels : ndarray
        Class labels (n_samples,)
    """
    n_samples, n_channels, n_timepoints = signals.shape
    # Extract features from first sample to determine feature dimension
    sample_features = feature_extraction.extract_all_features(
        signals[0], SAMPLING_RATE
    )
    n_features = len(sample_features)
    print(f"Extracting {n_features} features from {n_samples} samples...")
    print(f"Feature composition: {n_features} total features")
    # Pre-allocate feature array
    features = np.zeros((n_samples, n_features))
    # Extract features for each sample
    for i in range(n_samples):
        if i % 500 == 0:
            print(f"  Extracting features from sample {i}/{n_samples}")
        # Extract all features for this sample
        sample_features = feature_extraction.extract_all_features(
            signals[i], SAMPLING_RATE
        )
        features[i] = sample_features
    print("Feature extraction completed!")
    return features, labels
def validate_extracted_features(features, labels):
    """
    Validate extracted features
    Parameters:
    features : ndarray
        Extracted features (n_samples, n_features)
    labels : ndarray
        Class labels (n_samples,)
    """
    print("\n=== FEATURE EXTRACTION VALIDATION ===")
    # Check shapes
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    n_samples, n_features = features.shape
    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    if nan_count == 0:
        print("✓ No NaN values found in features")
    else:
        print(f"✗ Found {nan_count} NaN values in features")
    if inf_count == 0:
        print("✓ No Inf values found in features")
    else:
        print(f"✗ Found {inf_count} Inf values in features")
    # Check feature ranges
    feature_stds = np.std(features, axis=0)
    constant_features = np.where(feature_stds == 0)[0]
    print(f"\nNumber of constant features: {len(constant_features)}")
    feature_ranges = np.max(features, axis=0) - np.min(features, axis=0)
    large_range_features = np.where(feature_ranges > 1000)[0]
    print(f"Number of features with large ranges (>1000): {len(large_range_features)}")
    # Print statistics for a selection of features with reasonable ranges
    reasonable_features = [i for i in range(min(20, n_features))
                           if i not in constant_features and i not in large_range_features]
    print(f"\nFeature statistics (first {min(10, len(reasonable_features))} features with reasonable ranges):")
    for i in reasonable_features[:10]:
        feature_vals = features[:, i]
        print(f"  Feature {i:2d}: Min={np.min(feature_vals):8.4f}, "
              f"Max={np.max(feature_vals):8.4f}, "
              f"Mean={np.mean(feature_vals):8.4f}, "
              f"Std={np.std(feature_vals):8.4f}")
    # Check for class separability (simple per-class means for first few features)
    print("\nPer-class feature means (first 5 features with reasonable variance):")
    unique_labels = np.unique(labels)
    reasonable_feature_indices = [i for i in range(n_features)
                                  if i not in constant_features and i not in large_range_features][:5]
    for label in unique_labels:
        class_mask = labels == label
        class_features = features[class_mask]
        print(f"  Class {label}:")
        for i in reasonable_feature_indices:
            mean_val = np.mean(class_features[:, i])
            print(f"    Feature {i:2d}: {mean_val:8.4f}")
def save_extracted_features(features, labels, filepath):
    """
    Save extracted features to NPZ file
    Parameters:
    features : ndarray
        Extracted features (n_samples, n_features)
    labels : ndarray
        Class labels (n_samples,)
    filepath : str
        Path to save the features
    """
    np.savez_compressed(filepath, features=features, labels=labels)
    print(f"\nExtracted features saved to {filepath}")
def load_preprocessed_data(filepath):
    """
    Load preprocessed EEG data
    Parameters:
    filepath : str
        Path to the preprocessed data file
    Returns:
    signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    """
    data = np.load(filepath)
    signals = data['signals']
    labels = data['labels']
    print(f"Preprocessed data loaded from {filepath}")
    print(f"Signals shape: {signals.shape}")
    print(f"Labels shape: {labels.shape}")
    return signals, labels
if __name__ == "__main__":
    # Load preprocessed EEG data
    input_filepath = os.path.join(DATA_DIR, "preprocessed_eeg_data.npz")
    signals, labels = load_preprocessed_data(input_filepath)
    # Extract features
    features, feature_labels = extract_features_from_dataset(signals, labels)
    # Validate extracted features
    validate_extracted_features(features, feature_labels)
    # Save extracted features
    output_filepath = os.path.join(DATA_DIR, "extracted_features.npz")
    save_extracted_features(features, feature_labels, output_filepath)
    print("\nFeature extraction pipeline completed successfully!")
