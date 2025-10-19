"""
Train Random Forest baseline classifier for EEG command classification
"""
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report
import joblib
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR, CLASS_NAMES
def load_data():
    """
    Load training and validation feature datasets
    Returns:
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    X_val : ndarray
        Validation features
    y_val : ndarray
        Validation labels
    """
    # Load training data
    train_data_path = os.path.join(DATA_DIR, "features_train.npz")
    train_data = np.load(train_data_path)
    X_train = train_data['features']
    y_train = train_data['labels']
    print(f"Training data loaded: features shape {X_train.shape}, labels shape {y_train.shape}")
    # Load validation data
    val_data_path = os.path.join(DATA_DIR, "features_val.npz")
    val_data = np.load(val_data_path)
    X_val = val_data['features']
    y_val = val_data['labels']
    print(f"Validation data loaded: features shape {X_val.shape}, labels shape {y_val.shape}")
    return X_train, y_train, X_val, y_val
def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    Parameters:
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    Returns:
    rf_model : RandomForestClassifier
        Trained Random Forest model
    """
    print("\nTraining Random Forest classifier...")
    print("Parameters: n_estimators=100, max_depth=20, min_samples_split=5, random_state=42")
    # Initialize Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    # Train the model
    rf_model.fit(X_train, y_train)
    print("Training completed!")
    return rf_model
def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model on validation set
    Parameters:
    model : sklearn estimator
        Trained model
    X_val : ndarray
        Validation features
    y_val : ndarray
        Validation labels
    Returns:
    metrics : dict
        Dictionary containing evaluation metrics
    """
    print("\nEvaluating model on validation set...")
    # Make predictions
    y_pred = model.predict(X_val)
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
    # Confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    # Detailed classification report
    class_report = classification_report(y_val, y_pred, target_names=CLASS_NAMES, output_dict=True)
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
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")
    # Check if target accuracy is achieved
    if accuracy >= 0.60:
        print("✓ Target accuracy (≥60%) achieved!")
    else:
        print("✗ Target accuracy (≥60%) not achieved")
    return metrics, y_pred
def save_model(model, filepath):
    """
    Save trained model to disk
    Parameters:
    model : sklearn estimator
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
def save_metrics(metrics, filepath):
    """
    Save evaluation metrics to JSON file
    Parameters:
    metrics : dict
        Dictionary containing evaluation metrics
    filepath : str
        Path to save the metrics
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle nested dictionaries
            serializable_metrics[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    serializable_metrics[key][subkey] = subvalue.tolist()
                else:
                    serializable_metrics[key][subkey] = subvalue
        else:
            serializable_metrics[key] = value
    with open(filepath, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")
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
    plt.title('Confusion Matrix - Random Forest Baseline')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {filepath}")
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
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    # Evaluate model
    metrics, y_pred = evaluate_model(rf_model, X_val, y_val)
    # Print detailed metrics
    print_detailed_metrics(metrics)
    # Save model
    model_filepath = os.path.join(MODEL_DIR, "random_forest_baseline.pkl")
    save_model(rf_model, model_filepath)
    # Save metrics
    metrics_filepath = os.path.join(RESULTS_DIR, "rf_baseline_metrics.json")
    save_metrics(metrics, metrics_filepath)
    # Plot and save confusion matrix
    conf_matrix = np.array(metrics['confusion_matrix'])
    cm_plot_filepath = os.path.join(RESULTS_DIR, "confusion_matrix_rf.png")
    plot_confusion_matrix(conf_matrix, CLASS_NAMES, cm_plot_filepath)
    print("\nRandom Forest baseline training and evaluation completed successfully!")
