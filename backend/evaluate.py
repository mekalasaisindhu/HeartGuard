"""
HeartGuard - Model Evaluation Script
Generates a visual Confusion Matrix for the Hybrid Model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from train import load_dataset
from utils.preprocessing import TARGET_SR

MODELS_DIR = Path("backend/models")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Visualizes the confusion matrix using Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # Labeling the quadrants
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    
    output_path = MODELS_DIR / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Confusion matrix saved to: {output_path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, title='Receiver Operating Characteristic'):
    """
    Plots the ROC curve and calculates AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    output_path = MODELS_DIR / 'roc_curve.png'
    plt.savefig(output_path, dpi=150)
    print(f"✓ ROC curve saved to: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("HeartGuard - Hybrid Model Evaluation")
    print("=" * 60)

    # 1. Load trained models
    print("\nLoading models...")
    try:
        with open(MODELS_DIR / 'svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        cnn_model = keras.models.load_model(MODELS_DIR / 'cnn_model.h5')
        print("✓ Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Load test dataset
    try:
        X_features_test, X_images_test, y_test, _ = load_dataset('test')
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return

    # 3. Generate predictions
    print("\nGenerating predictions...")
    # SVM
    X_test_scaled = scaler.transform(X_features_test)
    svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    # CNN
    cnn_probs = cnn_model.predict(X_images_test, verbose=0).flatten()
    
    # Hybrid (40% SVM, 60% CNN) - Using same weights as train.py
    hybrid_probs = (0.4 * svm_probs) + (0.6 * cnn_probs)
    hybrid_preds = (hybrid_probs > 0.5).astype(int)

    # 4. Metrics
    acc = accuracy_score(y_test, hybrid_preds)
    print(f"\nHybrid Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, hybrid_preds, target_names=['Normal', 'Abnormal']))

    # 5. Visual Metrics
    cm = confusion_matrix(y_test, hybrid_preds)
    plot_confusion_matrix(cm, classes=['Normal', 'Abnormal'], title='HeartGuard Hybrid Model Confusion Matrix')
    
    plot_roc_curve(y_test, hybrid_probs, title='HeartGuard Hybrid Model ROC Curve')

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
