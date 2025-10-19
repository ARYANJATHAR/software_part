"""
Split EEG data into training and validation sets with stratification
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from config import DATA_DIR
def load_datasets():
    """
    Load both feature and preprocessed signal datasets
    Returns:
    feature_data : dict
        Features and labels from extracted_features.npz
    signal_data : dict
        Signals and labels from preprocessed_eeg_data.npz
    """
    # Load feature dataset
    feature_path = os.path.join(DATA_DIR, "extracted_features.npz")
    feature_data = np.load(feature_path)
    print(f"Feature data loaded from {feature_path}")
    print(f"Features shape: {feature_data['features'].shape}")
    print(f"Labels shape: {feature_data['labels'].shape}")
    # Load preprocessed signal dataset
    signal_path = os.path.join(DATA_DIR, "preprocessed_eeg_data.npz")
    signal_data = np.load(signal_path)
    print(f"Signal data loaded from {signal_path}")
    print(f"Signals shape: {signal_data['signals'].shape}")
    print(f"Labels shape: {signal_data['labels'].shape}")
    return feature_data, signal_data
def split_data_with_stratification(features, signals, labels, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets with stratification
    Parameters:
    features : ndarray
        Extracted features (n_samples, n_features)
    signals : ndarray
        Preprocessed EEG signals (n_samples, n_channels, n_timepoints)
    labels : ndarray
        Class labels (n_samples,)
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random state for reproducibility
    Returns:
    train_indices : ndarray
        Indices for training set
    val_indices : ndarray
        Indices for validation set
    """
    print(f"\nSplitting data with stratification (test_size={test_size})...")
    # Split indices only (to ensure both datasets use the same split)
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    print(f"Training set size: {len(train_indices)} samples")
    print(f"Validation set size: {len(val_indices)} samples")
    return train_indices, val_indices
def validate_splits(labels, train_indices, val_indices):
    """
    Validate the train/validation splits
    Parameters:
    labels : ndarray
        Class labels (n_samples,)
    train_indices : ndarray
        Indices for training set
    val_indices : ndarray
        Indices for validation set
    """
    print("\n=== SPLIT VALIDATION ===")
    # Check for overlap between train and validation sets
    overlap = np.intersect1d(train_indices, val_indices)
    if len(overlap) == 0:
        print("✓ No overlap between training and validation sets")
    else:
        print(f"✗ Found {len(overlap)} overlapping samples between train and validation sets")
    # Check class distribution in training set
    train_labels = labels[train_indices]
    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    print(f"\nTraining set class distribution:")
    for label, count in zip(unique_train_labels, train_counts):
        print(f"  Class {label}: {count} samples ({count / len(train_labels) * 100:.1f}%)")
    # Check class distribution in validation set
    val_labels = labels[val_indices]
    unique_val_labels, val_counts = np.unique(val_labels, return_counts=True)
    print(f"\nValidation set class distribution:")
    for label, count in zip(unique_val_labels, val_counts):
        print(f"  Class {label}: {count} samples ({count / len(val_labels) * 100:.1f}%)")
    # Check expected counts
    expected_train_size = int(len(labels) * 0.8)
    expected_val_size = len(labels) - expected_train_size
    print(f"\nSize validation:")
    print(f"  Expected training size: ~{expected_train_size}")
    print(f"  Actual training size: {len(train_indices)}")
    print(f"  Expected validation size: ~{expected_val_size}")
    print(f"  Actual validation size: {len(val_indices)}")
    # Check if sizes are within reasonable range
    if abs(len(train_indices) - expected_train_size) <= 10:
        print("✓ Training set size is within expected range")
    else:
        print("✗ Training set size differs significantly from expected")
    if abs(len(val_indices) - expected_val_size) <= 10:
        print("✓ Validation set size is within expected range")
    else:
        print("✗ Validation set size differs significantly from expected")
def save_split_indices(train_indices, val_indices, filepath):
    """
    Save train/validation indices for reproducibility
    Parameters:
    train_indices : ndarray
        Indices for training set
    val_indices : ndarray
        Indices for validation set
    filepath : str
        Path to save the indices
    """
    np.savez_compressed(filepath, train_indices=train_indices, val_indices=val_indices)
    print(f"\nSplit indices saved to {filepath}")
def create_split_datasets(feature_data, signal_data, train_indices, val_indices):
    """
    Create train/validation datasets for both features and signals
    Parameters:
    feature_data : dict
        Features and labels from extracted_features.npz
    signal_data : dict
        Signals and labels from preprocessed_eeg_data.npz
    train_indices : ndarray
        Indices for training set
    val_indices : ndarray
        Indices for validation set
    """
    # Create feature train/validation datasets
    feature_train_data = {
        'features': feature_data['features'][train_indices],
        'labels': feature_data['labels'][train_indices]
    }
    feature_val_data = {
        'features': feature_data['features'][val_indices],
        'labels': feature_data['labels'][val_indices]
    }
    # Create signal train/validation datasets
    signal_train_data = {
        'signals': signal_data['signals'][train_indices],
        'labels': signal_data['labels'][train_indices]
    }
    signal_val_data = {
        'signals': signal_data['signals'][val_indices],
        'labels': signal_data['labels'][val_indices]
    }
    # Save all split datasets
    feature_train_path = os.path.join(DATA_DIR, "features_train.npz")
    feature_val_path = os.path.join(DATA_DIR, "features_val.npz")
    signal_train_path = os.path.join(DATA_DIR, "signals_train.npz")
    signal_val_path = os.path.join(DATA_DIR, "signals_val.npz")
    np.savez_compressed(feature_train_path, **feature_train_data)
    np.savez_compressed(feature_val_path, **feature_val_data)
    np.savez_compressed(signal_train_path, **signal_train_data)
    np.savez_compressed(signal_val_path, **signal_val_data)
    print(f"\nSplit datasets saved:")
    print(f"  Feature train: {feature_train_path} (shape: {feature_train_data['features'].shape})")
    print(f"  Feature validation: {feature_val_path} (shape: {feature_val_data['features'].shape})")
    print(f"  Signal train: {signal_train_path} (shape: {signal_train_data['signals'].shape})")
    print(f"  Signal validation: {signal_val_path} (shape: {signal_val_data['signals'].shape})")
if __name__ == "__main__":
    # Load datasets
    feature_data, signal_data = load_datasets()
    # Get labels (should be the same for both datasets)
    labels = feature_data['labels']
    # Split data with stratification
    train_indices, val_indices = split_data_with_stratification(
        feature_data['features'],
        signal_data['signals'],
        labels
    )
    # Validate splits
    validate_splits(labels, train_indices, val_indices)
    # Save split indices for reproducibility
    indices_filepath = os.path.join(DATA_DIR, "train_val_split.npz")
    save_split_indices(train_indices, val_indices, indices_filepath)
    # Create and save split datasets
    create_split_datasets(feature_data, signal_data, train_indices, val_indices)
    print("\nData splitting completed successfully!")
