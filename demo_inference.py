"""
Demonstration script for end-to-end EEG inference with visualization
This script showcases the complete inference pipeline with both Random Forest
and CNN models, comparing their predictions and visualizing results.
"""
import numpy as np
import time
import argparse
import sys
import os
# Import our modules (repo-relative configuration)
from config import (
    DATA_DIR,
    RESULTS_DIR,
    CLASS_LABELS,
    CLASS_NAMES,
)
from inference import RFInferenceEngine
from inference_cnn import CNNInferenceEngine
from visualize_predictions import EEGVisualizer
def load_validation_data():
    """Load validation data for demonstration"""
    print("Loading validation data...")
    val_data = np.load(os.path.join(DATA_DIR, 'signals_val.npz'))
    signals = val_data['signals']
    labels = val_data['labels']
    print(f"Loaded {signals.shape[0]} validation samples")
    return signals, labels
def simulate_eeg_signal(command_class=0, duration=2, sampling_rate=250, n_channels=8):
    """
    Generate simulated EEG signal for a specific command class
    Args:
        command_class (int): Class index (0-4)
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
    # Base frequencies for different classes
    base_freqs = [10, 15, 20, 25, 30]  # Different base frequencies for each class
    base_freq = base_freqs[command_class % len(base_freqs)]
    for ch in range(n_channels):
        # Create signal with base frequency and harmonics
        signal_data[ch] = (
            np.sin(2 * np.pi * base_freq * t) +  # Base frequency component
            0.5 * np.sin(2 * np.pi * (base_freq * 1.5) * t) +  # Harmonic
            0.3 * np.sin(2 * np.pi * (base_freq * 2) * t) +  # Second harmonic
            0.1 * np.random.randn(n_samples) * (1 + 0.5 * ch)  # Channel-specific noise
        )
    return signal_data
def demonstrate_models(signals, labels, num_samples=5):
    """Demonstrate both models on validation data"""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Comparing Random Forest and CNN Models")
    print("=" * 60)
    # Initialize inference engines
    print("Initializing inference engines...")
    rf_engine = RFInferenceEngine()
    cnn_engine = CNNInferenceEngine()
    cnn_engine.warmup()  # Warm up for consistent timing
    # Initialize visualizer
    visualizer = EEGVisualizer()
    # Track performance
    rf_correct = 0
    cnn_correct = 0
    rf_times = []
    cnn_times = []
    # History for visualization
    rf_history_labels = []
    rf_history_confidences = []
    cnn_history_labels = []
    cnn_history_confidences = []
    print(f"\nTesting on {num_samples} validation samples...")
    print("-" * 60)
    for i in range(min(num_samples, len(signals))):
        # Get sample
        signal_sample = signals[i]
        true_label_idx = labels[i]
        true_label = CLASS_LABELS[true_label_idx]
        print(f"\nSample {i + 1}/{num_samples}: True Label = {true_label}")
        print("-" * 40)
        # Random Forest inference
        start_time = time.time()
        rf_pred_label, rf_probabilities = rf_engine.predict(signal_sample, return_proba=True)
        rf_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        rf_confidence = np.max(rf_probabilities)
        rf_times.append(rf_time)
        # Check RF accuracy
        if CLASS_NAMES.index(rf_pred_label) == true_label_idx:
            rf_correct += 1
            rf_status = "✓ CORRECT"
        else:
            rf_status = "✗ INCORRECT"
        # CNN inference
        start_time = time.time()
        cnn_pred_label, cnn_probabilities = cnn_engine.predict(signal_sample, return_proba=True)
        cnn_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        cnn_confidence = np.max(cnn_probabilities)
        cnn_times.append(cnn_time)
        # Check CNN accuracy
        if CLASS_NAMES.index(cnn_pred_label) == true_label_idx:
            cnn_correct += 1
            cnn_status = "✓ CORRECT"
        else:
            cnn_status = "✗ INCORRECT"
        # Print results
        print(f"Random Forest: {rf_pred_label} (Conf: {rf_confidence:.3f}, Time: {rf_time:.1f}ms) {rf_status}")
        print(f"CNN Model:     {cnn_pred_label} (Conf: {cnn_confidence:.3f}, Time: {cnn_time:.1f}ms) {cnn_status}")
        # Update history for visualization
        rf_history_labels.append(rf_pred_label)
        rf_history_confidences.append(rf_confidence)
        cnn_history_labels.append(cnn_pred_label)
        cnn_history_confidences.append(cnn_confidence)
        # Visualize for the last sample
        if i == min(num_samples, len(signals)) - 1:
            print(f"\nGenerating visualization for sample {i + 1}...")
            # Update visualization with Random Forest results
            visualizer.update_visualization(
                signal_sample,
                rf_probabilities,
                rf_pred_label,
                rf_confidence,
                rf_history_labels,
                rf_history_confidences,
                channel=0
            )
            # Save visualization
            rf_viz_path = os.path.join(RESULTS_DIR, f"rf_demo_visualization_{i + 1}.png")
            visualizer.save_visualization(rf_viz_path)
            print(f"Random Forest visualization saved to {rf_viz_path}")
            # Update visualization with CNN results
            visualizer.update_visualization(
                signal_sample,
                cnn_probabilities,
                cnn_pred_label,
                cnn_confidence,
                cnn_history_labels,
                cnn_history_confidences,
                channel=0
            )
            # Save visualization
            cnn_viz_path = os.path.join(RESULTS_DIR, f"cnn_demo_visualization_{i + 1}.png")
            visualizer.save_visualization(cnn_viz_path)
            print(f"CNN visualization saved to {cnn_viz_path}")
    # Print summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    rf_accuracy = rf_correct / num_samples * 100
    cnn_accuracy = cnn_correct / num_samples * 100
    rf_avg_time = np.mean(rf_times)
    cnn_avg_time = np.mean(cnn_times)
    print(f"Random Forest - Accuracy: {rf_accuracy:.1f}% ({rf_correct}/{num_samples})")
    print(f"                Avg Time: {rf_avg_time:.1f}ms per sample")
    print(f"CNN Model     - Accuracy: {cnn_accuracy:.1f}% ({cnn_correct}/{num_samples})")
    print(f"                Avg Time: {cnn_avg_time:.1f}ms per sample")
    print(f"\nPerformance Ratio (RF/CNN): {rf_avg_time / cnn_avg_time:.1f}x latency")
    print(f"Accuracy Difference: {rf_accuracy - cnn_accuracy:+.1f}%")
def demonstrate_simulated_signals():
    """Demonstrate models on simulated signals for each command class"""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Simulated EEG Signals for Each Command Class")
    print("=" * 60)
    # Initialize inference engines
    rf_engine = RFInferenceEngine()
    cnn_engine = CNNInferenceEngine()
    cnn_engine.warmup()
    print("Testing models on simulated signals for each command class:")
    print("-" * 60)
    for class_idx, class_name in CLASS_LABELS.items():
        print(f"\nTesting {class_name.upper()} command simulation:")
        # Generate simulated signal
        signal = simulate_eeg_signal(command_class=class_idx)
        # Random Forest prediction
        rf_pred, rf_proba = rf_engine.predict(signal, return_proba=True)
        rf_conf = np.max(rf_proba)
        # CNN prediction
        cnn_pred, cnn_proba = cnn_engine.predict(signal, return_proba=True)
        cnn_conf = np.max(cnn_proba)
        print(f"  Random Forest: {rf_pred} (Confidence: {rf_conf:.3f})")
        print(f"  CNN Model:     {cnn_pred} (Confidence: {cnn_conf:.3f})")
def demonstrate_batch_processing(signals, batch_size=8):
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print(f"DEMONSTRATION: Batch Processing (Batch Size: {batch_size})")
    print("=" * 60)
    # Initialize CNN engine for batch processing
    cnn_engine = CNNInferenceEngine()
    cnn_engine.warmup()
    # Prepare batch
    batch_signals = signals[:batch_size]
    print(f"Processing batch of {batch_size} samples...")
    # Individual processing
    start_time = time.time()
    individual_results = []
    for signal in batch_signals:
        # Fix: cnn_engine.predict returns single value when return_proba=False
        pred = cnn_engine.predict(signal, return_proba=False)
        individual_results.append(pred)
    individual_time = (time.time() - start_time) * 1000
    # Batch processing
    start_time = time.time()
    batch_results = cnn_engine.process_batch(batch_signals, return_proba=False)
    batch_time = (time.time() - start_time) * 1000
    print(f"Individual processing: {individual_time:.1f}ms ({individual_time / batch_size:.2f}ms per sample)")
    print(f"Batch processing:      {batch_time:.1f}ms ({batch_time / batch_size:.2f}ms per sample)")
    print(f"Speedup:               {individual_time / batch_time:.1f}x faster")
    print("\nSample predictions:")
    for i, (ind_pred, batch_pred) in enumerate(zip(individual_results, batch_results)):
        status = "✓ MATCH" if ind_pred == batch_pred else "✗ DIFFERENT"
        print(f"  Sample {i + 1}: Individual={ind_pred}, Batch={batch_pred} {status}")
def main():
    """Main demonstration function"""
    print("EEG Command Classifier - End-to-End Demonstration")
    print("=" * 50)
    # Load validation data
    signals, labels = load_validation_data()
    # Run demonstrations
    demonstrate_models(signals, labels, num_samples=10)
    demonstrate_simulated_signals()
    demonstrate_batch_processing(signals, batch_size=8)
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Check the results/ directory for generated visualizations.")
    print("Try running individual inference scripts for more detailed testing:")
    print("  python inference.py     # Random Forest inference")
    print("  python inference_cnn.py # CNN inference")
if __name__ == "__main__":
    main()
