"""
Visualization module for real-time EEG predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import os
import sys
from config import (
    CLASS_NAMES,
    RESULTS_DIR,
)
# Set up matplotlib style
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 10
class EEGVisualizer:
    """Visualization module for EEG signals and predictions"""
    def __init__(self, sampling_rate=250, n_channels=8):
        """
        Initialize the visualizer
        Parameters:
        sampling_rate (int): Sampling rate in Hz
        n_channels (int): Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.time_points = np.linspace(0, 2, sampling_rate * 2)  # 2 seconds of data
        self.class_names = CLASS_NAMES
        # Setup figure and subplots
        self.fig = plt.figure(figsize=(15, 10))
        # Subplot for EEG signals (first 8 channels)
        self.ax_signals = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        # Subplot for prediction probabilities
        self.ax_probs = plt.subplot2grid((3, 4), (0, 3))
        # Subplot for prediction history/timeline
        self.ax_timeline = plt.subplot2grid((3, 4), (1, 3))
        # Subplot for frequency spectrum
        self.ax_spectrum = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        # Subplot for confusion matrix comparison
        self.ax_confusion = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        plt.tight_layout()
    def plot_eeg_signals(self, signal_data, title="EEG Signals"):
        """
        Plot multi-channel EEG signals
        Parameters:
        signal_data (np.ndarray): EEG signal of shape (n_channels, n_samples)
        title (str): Plot title
        """
        self.ax_signals.clear()
        # Plot each channel with offset for clarity
        for ch in range(min(self.n_channels, signal_data.shape[0])):
            channel_data = signal_data[ch]
            # Normalize channel data for better visualization
            if np.std(channel_data) > 0:
                normalized_data = (channel_data - np.mean(channel_data)) / np.std(channel_data)
            else:
                normalized_data = channel_data
            # Offset each channel for clarity
            offset_data = normalized_data + ch * 3
            self.ax_signals.plot(self.time_points[:len(channel_data)], offset_data,
                                 label=f'Channel {ch + 1}', linewidth=0.8)
        self.ax_signals.set_title(title)
        self.ax_signals.set_xlabel('Time (s)')
        self.ax_signals.set_ylabel('Amplitude (offset)')
        self.ax_signals.legend(loc='upper right', fontsize=6)
        self.ax_signals.grid(True, alpha=0.3)
    def plot_prediction_probabilities(self, probabilities, pred_label=None):
        """
        Plot prediction probabilities as bar chart
        Parameters:
        probabilities (np.ndarray): Prediction probabilities for each class
        pred_label (str): Predicted label for highlighting
        """
        self.ax_probs.clear()
        colors = ['lightblue'] * len(self.class_names)
        if pred_label and pred_label in self.class_names:
            pred_idx = self.class_names.index(pred_label)
            colors[pred_idx] = 'orange'  # Highlight predicted class
        bars = self.ax_probs.bar(range(len(self.class_names)), probabilities, color=colors)
        self.ax_probs.set_xticks(range(len(self.class_names)))
        self.ax_probs.set_xticklabels(self.class_names, rotation=45, ha='right')
        self.ax_probs.set_ylabel('Probability')
        self.ax_probs.set_title('Prediction Probabilities')
        self.ax_probs.set_ylim(0, 1)
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            self.ax_probs.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                               f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
    def plot_prediction_timeline(self, history_labels, history_confidences, max_history=20):
        """
        Plot timeline of recent predictions
        Parameters:
        history_labels (list): List of recent prediction labels
        history_confidences (list): List of recent prediction confidences
        max_history (int): Maximum number of history items to display
        """
        self.ax_timeline.clear()
        # Limit history to max_history items
        if len(history_labels) > max_history:
            history_labels = history_labels[-max_history:]
            history_confidences = history_confidences[-max_history:]
        # Create color mapping based on confidence
        colors = []
        for conf in history_confidences:
            if conf > 0.8:
                colors.append('green')      # High confidence
            elif conf > 0.5:
                colors.append('orange')     # Medium confidence
            else:
                colors.append('red')        # Low confidence
        # Plot timeline
        indices = range(len(history_labels))
        self.ax_timeline.scatter(indices, [1] * len(history_labels),
                                 c=colors, s=50, alpha=0.7)
        # Add labels
        for i, (label, conf) in enumerate(zip(history_labels, history_confidences)):
            self.ax_timeline.text(i, 1.05, f'{label[:3]}\n{conf:.2f}',
                                  ha='center', va='bottom', fontsize=6)
        self.ax_timeline.set_ylim(0.9, 1.2)
        self.ax_timeline.set_xlim(-0.5, len(history_labels) - 0.5)
        self.ax_timeline.set_yticks([])
        self.ax_timeline.set_title('Prediction History')
        self.ax_timeline.set_xlabel('Recent Predictions')
    def plot_frequency_spectrum(self, signal_data, channel=0):
        """
        Plot frequency spectrum of EEG signal
        Parameters:
        signal_data (np.ndarray): EEG signal of shape (n_channels, n_samples)
        channel (int): Channel to analyze
        """
        self.ax_spectrum.clear()
        if channel >= signal_data.shape[0]:
            channel = 0
        # Get channel data
        channel_data = signal_data[channel]
        # Compute FFT
        fft_vals = np.fft.fft(channel_data)
        fft_freq = np.fft.fftfreq(len(channel_data), 1 / self.sampling_rate)
        # Only use positive frequencies
        positive_freq_idx = fft_freq >= 0
        fft_vals = fft_vals[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]
        # Power spectral density
        psd = np.abs(fft_vals) ** 2
        # Plot
        self.ax_spectrum.plot(fft_freq, psd, linewidth=1)
        self.ax_spectrum.set_title(f'Frequency Spectrum (Channel {channel + 1})')
        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.set_ylabel('Power')
        self.ax_spectrum.grid(True, alpha=0.3)
        # Highlight frequency bands
        bands = {
            'δ (0.5-4 Hz)': (0.5, 4, 'purple'),
            'θ (4-8 Hz)': (4, 8, 'blue'),
            'α (8-13 Hz)': (8, 13, 'green'),
            'β (13-30 Hz)': (13, 30, 'orange'),
            'γ (30-45 Hz)': (30, 45, 'red')
        }
        for band_name, (low, high, color) in bands.items():
            if low <= 30:  # Only show bands up to 30 Hz for clarity
                self.ax_spectrum.axvspan(low, high, alpha=0.2, color=color,
                                         label=f'{band_name}')
        self.ax_spectrum.legend(loc='upper right', fontsize=6)
    def plot_confusion_matrix_comparison(self, conf_matrix_path=None):
        """
        Plot confusion matrix comparison (from validation results)
        Parameters:
        conf_matrix_path (str): Path to confusion matrix image
        """
        self.ax_confusion.clear()
        # If confusion matrix image exists, load and display it
        if conf_matrix_path and os.path.exists(conf_matrix_path):
            try:
                from PIL import Image
                img = Image.open(conf_matrix_path)
                self.ax_confusion.imshow(img)
                self.ax_confusion.axis('off')
                self.ax_confusion.set_title('Confusion Matrix (Validation)')
                return
            except:
                pass  # Fall back to placeholder if image loading fails
        # Placeholder text if no image available
        self.ax_confusion.text(0.5, 0.5, 'Confusion Matrix\n(Validation Results)',
                               ha='center', va='center', transform=self.ax_confusion.transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.ax_confusion.axis('off')
        self.ax_confusion.set_title('Confusion Matrix (Validation)')
    def update_visualization(self, signal_data, probabilities, pred_label, confidence,
                             history_labels, history_confidences, channel=0):
        """
        Update all visualization components
        Parameters:
        signal_data (np.ndarray): EEG signal of shape (n_channels, n_samples)
        probabilities (np.ndarray): Prediction probabilities
        pred_label (str): Predicted label
        confidence (float): Prediction confidence
        history_labels (list): History of predictions
        history_confidences (list): History of confidences
        channel (int): Channel for frequency analysis
        """
        # Update all plots
        self.plot_eeg_signals(signal_data, f"EEG Signals (Prediction: {pred_label}, Confidence: {confidence:.2f})")
        self.plot_prediction_probabilities(probabilities, pred_label)
        self.plot_prediction_timeline(history_labels, history_confidences)
        self.plot_frequency_spectrum(signal_data, channel)
        self.plot_confusion_matrix_comparison(os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png"))
        plt.draw()
    def save_visualization(self, filepath):
        """
        Save current visualization to file
        Parameters:
        filepath (str): Path to save visualization
        """
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filepath}")
    def show(self):
        """Display the visualization"""
        plt.show()
def demonstrate_visualization():
    """Demonstrate the visualization module with sample data"""
    print("Demonstrating EEG visualization module...")
    # Initialize visualizer
    visualizer = EEGVisualizer()
    # Generate sample EEG data
    n_samples = 500
    n_channels = 8
    t = np.linspace(0, 2, n_samples)
    # Create sample signal with multiple components
    signal_data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        signal_data[ch] = (
            np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
            0.5 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz component
            0.3 * np.sin(2 * np.pi * 30 * t) +  # 30 Hz component
            0.1 * np.random.randn(n_samples)  # Noise
        )
    # Sample predictions
    probabilities = np.array([0.1, 0.2, 0.05, 0.5, 0.15])  # Sample probabilities
    pred_label = "light_on"
    confidence = 0.5
    # Sample history
    history_labels = ["neutral", "fan_on", "fan_off", "light_on", "neutral"]
    history_confidences = [0.9, 0.8, 0.7, 0.95, 0.85]
    # Update visualization
    visualizer.update_visualization(
        signal_data,
        probabilities,
        pred_label,
        confidence,
        history_labels,
        history_confidences
    )
    # Save visualization
    output_path = os.path.join(RESULTS_DIR, "sample_visualization.png")
    visualizer.save_visualization(output_path)
    print(f"Sample visualization saved to {output_path}")
    print("Visualization demonstration completed!")
if __name__ == "__main__":
    demonstrate_visualization()
