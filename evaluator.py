import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd

from config import (PLOTS_DIR, METRICS_DIR, CONFUSION_MATRIX_FILE, ROC_CURVE_FILE, PR_CURVE_FILE,
                    TRAINING_HISTORY_FILE, CLASSIFICATION_REPORT_FILE, MODEL_SUMMARY_FILE)


def plot_training_history(history, save_path=TRAINING_HISTORY_FILE):
    if not history or not history.history:
        print("No training history to plot.")
        return

    history_df = pd.DataFrame(history.history)
    
    plt.figure(figsize=(12, 5))
    
    if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        plt.subplot(1, 2, 1)
        plt.plot(history_df['accuracy'], label='Train Accuracy')
        plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(history_df['loss'], label='Train Loss')
        plt.plot(history_df['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred_classes, save_path=CONFUSION_MATRIX_FILE):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Churn', 'Predicted Churn'],
                yticklabels=['Actual No Churn', 'Actual Churn'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def plot_roc_curve(y_true, y_pred_probs, save_path=ROC_CURVE_FILE):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve plot saved to {save_path}")

def plot_precision_recall_curve(y_true, y_pred_probs, save_path=PR_CURVE_FILE):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.7, where='post', label=f'AP = {avg_precision:.2f}')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(save_path)
    plt.close()
    print(f"Precision-Recall curve plot saved to {save_path}")

def save_classification_report(y_true, y_pred_classes, save_path=CLASSIFICATION_REPORT_FILE):
    report = classification_report(y_true, y_pred_classes, target_names=['No Churn', 'Churn'])
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")
    print("Classification Report:", report)

def save_model_summary_text(model, save_path=MODEL_SUMMARY_FILE):
    if hasattr(model, 'summary'):
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = "".join(summary_list)
        with open(save_path, 'w') as f:
            f.write(summary_str)
        print(f"Model summary saved to {save_path}")


def generate_evaluation_dashboard(model, history, X_test, y_test):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Plot training history
    plot_training_history(history)

    # Get predictions
    y_pred_probs = model.predict(X_test).ravel() # Probabilities
    y_pred_classes = (y_pred_probs > 0.5).astype(int) # Binary classes

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes)

    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_probs)
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_test, y_pred_probs)

    # Save classification report
    save_classification_report(y_test, y_pred_classes)
    
    # Save model summary
    save_model_summary_text(model)


if __name__ == '__main__':
    # Mock data for testing evaluator
    class MockModel:
        def predict(self, X):
            return np.random.rand(len(X), 1) # Simulate probability outputs
        def summary(self, print_fn=None):
            summary_lines = ["Layer (type)                 Output Shape              Param #   ",
                             "=================================================================",
                             "dense (Dense)                (None, 64)                704       ",
                             "dropout (Dropout)            (None, 64)                0         ",
                             "dense_1 (Dense)              (None, 1)                 65        ",
                             "=================================================================",
                             "Total params: 769", "Trainable params: 769", "Non-trainable params: 0"]
            if print_fn:
                for line in summary_lines:
                    print_fn(line)
            else: # Default print_fn for model.summary() is print
                 for line in summary_lines:
                    print(line)


    class MockHistory:
        def __init__(self):
            self.history = {
                'loss': [0.5, 0.4, 0.3],
                'accuracy': [0.7, 0.75, 0.8],
                'val_loss': [0.45, 0.35, 0.25],
                'val_accuracy': [0.72, 0.77, 0.82]
            }

    mock_model_instance = MockModel()
    mock_history_instance = MockHistory()
    
    X_test_sample = pd.DataFrame(np.random.rand(50, 10))
    y_test_sample = pd.Series(np.random.randint(0, 2, 50))
    
    print("Generating mock evaluation dashboard...")
    generate_evaluation_dashboard(mock_model_instance, mock_history_instance, X_test_sample, y_test_sample)
    print("Mock dashboard generation complete. Check 'output' directory.")