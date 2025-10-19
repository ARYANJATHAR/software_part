"""
Train 1D CNN model for EEG command classification
"""
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, CLASS_NAMES, N_CLASSES
class EEGDataset(Dataset):
    """
    Dataset class for EEG signals
    """
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
class EEG1DCNN(nn.Module):
    """
    1D CNN model for EEG classification
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
def load_data():
    """
    Load training and validation signal datasets
    Returns:
    train_dataset : EEGDataset
        Training dataset
    val_dataset : EEGDataset
        Validation dataset
    """
    # Load training data
    train_data_path = os.path.join(DATA_DIR, "signals_train.npz")
    train_data = np.load(train_data_path)
    signals_train = train_data['signals']
    labels_train = train_data['labels']
    print(f"Training data loaded: signals shape {signals_train.shape}, labels shape {labels_train.shape}")
    # Load validation data
    val_data_path = os.path.join(DATA_DIR, "signals_val.npz")
    val_data = np.load(val_data_path)
    signals_val = val_data['signals']
    labels_val = val_data['labels']
    print(f"Validation data loaded: signals shape {signals_val.shape}, labels shape {labels_val.shape}")
    # Create datasets
    train_dataset = EEGDataset(signals_train, labels_train)
    val_dataset = EEGDataset(signals_val, labels_val)
    return train_dataset, val_dataset
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10):
    """
    Train the CNN model
    Parameters:
    model : nn.Module
        CNN model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    device : torch.device
        Device to train on
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    patience : int
        Early stopping patience
    Returns:
    history : dict
        Training history
    best_model_state : dict
        Best model state dict
    """
    print(f"\nTraining CNN model on {device}")
    print(f"Parameters: epochs={epochs}, lr={lr}, patience={patience}")
    # Move model to device
    model.to(device)
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_epoch': 0
    }
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_acc)
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    print("Training completed!")
    return history, best_model_state
def evaluate_model(model, val_loader, device):
    """
    Evaluate the trained model on validation set
    Parameters:
    model : nn.Module
        Trained CNN model
    val_loader : DataLoader
        Validation data loader
    device : torch.device
        Device to evaluate on
    Returns:
    metrics : dict
        Dictionary containing evaluation metrics
    y_true : list
        True labels
    y_pred : list
        Predicted labels
    """
    print("\nEvaluating model on validation set...")
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_proba.extend(probabilities.cpu().numpy())
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        'accuracy': float(accuracy),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': conf_matrix.tolist()
    }
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")
    # Check if target accuracy is achieved
    if accuracy >= 0.70:
        print("✓ Target accuracy (≥70%) achieved!")
    else:
        print("✗ Target accuracy (≥70%) not achieved")
    # Check if all F1 scores meet target
    all_f1_above_target = all(f1_score >= 0.60 for f1_score in f1)
    if all_f1_above_target:
        print("✓ All per-class F1-scores (≥0.60) achieved!")
    else:
        print("✗ Some per-class F1-scores are below target (≥0.60)")
        for i, f1_score in enumerate(f1):
            if f1_score < 0.60:
                print(f"  Class {i} ({CLASS_NAMES[i]}): F1 = {f1_score:.4f}")
    return metrics, y_true, y_pred
def save_model(model_state, filepath):
    """
    Save trained model state to disk
    Parameters:
    model_state : dict
        Model state dictionary
    filepath : str
        Path to save the model
    """
    torch.save(model_state, filepath)
    print(f"Model saved to {filepath}")
def save_history(history, filepath):
    """
    Save training history to JSON file
    Parameters:
    history : dict
        Training history
    filepath : str
        Path to save the history
    """
    # Convert numpy arrays and other non-serializable objects to serializable formats
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, (list, tuple)):
            serializable_history[key] = [float(v) if not isinstance(v, (int, str)) else v for v in value]
        else:
            serializable_history[key] = value
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to {filepath}")
def plot_confusion_matrix(conf_matrix, class_names, filepath):
    """
    Plot and save confusion matrix
    Parameters:
    conf_matrix : ndarray
        Confusion matrix
    class_names : list
        List of class names
    filepath : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - 1D CNN Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {filepath}")
def plot_training_history(history, filepath):
    """
    Plot and save training history
    Parameters:
    history : dict
        Training history
    filepath : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    # Plot accuracy
    ax2.plot(history['train_accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {filepath}")
def print_detailed_metrics(metrics):
    """
    Print detailed evaluation metrics
    Parameters:
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print("\n=== DETAILED EVALUATION METRICS ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1-Score: {metrics['f1_per_class'][i]:.4f}")
if __name__ == "__main__":
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU (no GPU available)")
    # Load data
    train_dataset, val_dataset = load_data()
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    # Initialize model
    model = EEG1DCNN(n_channels=8, n_timepoints=500, n_classes=N_CLASSES)
    print(f"\nModel architecture:")
    print(f"  Conv1D(8→32, kernel=5) → ReLU → MaxPool(2)")
    print(f"  Conv1D(32→64, kernel=5) → ReLU → MaxPool(2)")
    print(f"  Flatten → Dense(8000→128) → Dropout(0.5) → Dense(128→5)")
    # Train model
    history, best_model_state = train_model(
        model, train_loader, val_loader, device,
        epochs=50, lr=0.001, patience=10
    )
    # Load best model state for evaluation
    model.load_state_dict(best_model_state)
    # Evaluate model
    metrics, y_true, y_pred = evaluate_model(model, val_loader, device)
    # Print detailed metrics
    print_detailed_metrics(metrics)
    # Save model
    model_filepath = os.path.join(MODEL_DIR, "cnn_model.pth")
    save_model(best_model_state, model_filepath)
    # Save training history
    history_filepath = os.path.join(RESULTS_DIR, "cnn_training_history.json")
    save_history(history, history_filepath)
    # Plot and save confusion matrix
    conf_matrix = np.array(metrics['confusion_matrix'])
    cm_plot_filepath = os.path.join(RESULTS_DIR, "confusion_matrix_cnn.png")
    plot_confusion_matrix(conf_matrix, CLASS_NAMES, cm_plot_filepath)
    # Plot and save training history
    history_plot_filepath = os.path.join(RESULTS_DIR, "cnn_training_history.png")
    plot_training_history(history, history_plot_filepath)
    print("\n1D CNN model training and evaluation completed successfully!")
