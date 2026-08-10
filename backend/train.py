"""
HeartGuard - Model Training Script
Trains SVM and CNN models for heart sound classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.preprocessing import preprocess_audio, mel_spectrogram_to_image
from utils.features import extract_all_features


# Configuration
DATASET_DIR = Path("dataset")
MODELS_DIR = Path("backend/models")
RANDOM_STATE = 42
TARGET_SIZE = (128, 128)


def load_dataset(split='train'):
    """
    Load dataset and extract features.
    
    Args:
        split: 'train' or 'test'
        
    Returns:
        X_features: Feature vectors for SVM
        X_images: Spectrogram images for CNN
        y: Labels (0=normal, 1=abnormal)
        file_paths: List of file paths
    """
    print(f"\nLoading {split} dataset...")
    
    split_dir = DATASET_DIR / split
    
    # Get all audio files
    normal_files = list((split_dir / 'normal').glob('*.wav'))
    abnormal_files = list((split_dir / 'abnormal').glob('*.wav'))
    
    print(f"  Normal files: {len(normal_files)}")
    print(f"  Abnormal files: {len(abnormal_files)}")
    
    # Combine files and labels
    files = normal_files + abnormal_files
    labels = [0] * len(normal_files) + [1] * len(abnormal_files)
    
    # Extract features and images
    features_list = []
    images_list = []
    valid_labels = []
    valid_files = []
    
    print(f"  Extracting features from {len(files)} files...")
    for file_path, label in tqdm(zip(files, labels), total=len(files)):
        try:
            # Preprocess audio
            preprocessed = preprocess_audio(str(file_path))
            audio = preprocessed['filtered_audio']
            mel_spec = preprocessed['mel_spectrogram']
            sr = preprocessed['sr']
            
            # Extract features for SVM
            features = extract_all_features(audio, sr)
            
            # Validate features (ensure it's a 1D array)
            if not isinstance(features, np.ndarray) or features.ndim != 1:
                print(f"  Warning: Invalid features for {file_path}, skipping...")
                continue
            
            features_list.append(features)
            
            # Convert spectrogram to image for CNN
            image = mel_spectrogram_to_image(mel_spec, target_size=TARGET_SIZE)
            
            # Validate image
            if not isinstance(image, np.ndarray) or image.shape != (*TARGET_SIZE, 1):
                print(f"  Warning: Invalid image for {file_path}, skipping...")
                continue
            
            images_list.append(image)
            
            valid_labels.append(label)
            valid_files.append(str(file_path))
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    # Check if we have any valid samples
    if len(features_list) == 0:
        raise ValueError(f"No valid samples loaded from {split} dataset! All files failed to process.")
    
    X_features = np.array(features_list)
    X_images = np.array(images_list)
    y = np.array(valid_labels)
    
    print(f"  ✓ Loaded {len(y)} samples")
    print(f"    Feature shape: {X_features.shape}")
    print(f"    Image shape: {X_images.shape}")
    
    return X_features, X_images, y, valid_files


def train_svm(X_train, y_train, X_test, y_test):
    """
    Train SVM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        model: Trained SVM model
        scaler: Feature scaler
    """
    print("\n" + "=" * 60)
    print("Training SVM Model")
    print("=" * 60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights to handle imbalance
    print("Calculating class weights...")
    print(f"  Class distribution: Normal={np.sum(y_train == 0)}, Abnormal={np.sum(y_train == 1)}")
    
    # Train SVM with RBF kernel and class weights
    print("Training SVM with RBF kernel and balanced class weights...")
    model = SVC(kernel='rbf', C=10, gamma='scale', 
                class_weight='balanced',  # Automatically handle class imbalance
                random_state=RANDOM_STATE, probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n✓ SVM Training Complete")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    print("\nTest Set Metrics:")
    print(f"  Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, test_pred):.4f}")
    print(f"  F1-score: {f1_score(y_test, test_pred):.4f}")
    
    # ROC-AUC score
    test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, test_pred_proba)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(f"  True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    return model, scaler


def build_cnn_model(input_shape=(128, 128, 1)):
    """
    Build CNN architecture.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        model: Compiled Keras model
    """
    print("Building CNN model...")
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def train_cnn(X_train, y_train, X_test, y_test):
    """
    Train CNN model.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
        
    Returns:
        model: Trained CNN model
        history: Training history
    """
    print("\n" + "=" * 60)
    print("Training CNN Model")
    print("=" * 60)
    
    # Build model
    model = build_cnn_model(input_shape=X_train.shape[1:])
    
    print("\nModel Architecture:")
    model.summary()
    
    # Calculate class weights for CNN
    print("\nCalculating class weights for CNN...")
    class_weights_array = compute_class_weight('balanced', 
                                                classes=np.unique(y_train), 
                                                y=y_train)
    class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"  Class weights: Normal={class_weight_dict[0]:.2f}, Abnormal={class_weight_dict[1]:.2f}")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        str(MODELS_DIR / 'cnn_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model with class weights
    print("\nTraining CNN with balanced class weights...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,  # Apply class weights
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Evaluate
    train_loss, train_acc, train_prec, train_rec = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n✓ CNN Training Complete")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    test_pred_prob = model.predict(X_test, verbose=0)
    test_pred = (test_pred_prob > 0.5).astype(int).flatten()
    
    print("\nTest Set Metrics:")
    print(f"  Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, test_pred):.4f}")
    print(f"  F1-score: {f1_score(y_test, test_pred):.4f}")
    
    # ROC-AUC score
    roc_auc = roc_auc_score(y_test, test_pred_prob)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(f"  True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    print("\nPer-Class Performance:")
    print(classification_report(y_test, test_pred, target_names=['Normal', 'Abnormal']))
    
    return model, history


def evaluate_hybrid(svm_model, scaler, cnn_model, X_features_test, X_images_test, y_test, svm_weight=0.4, cnn_weight=0.6):
    """
    Evaluate hybrid prediction.
    
    Args:
        svm_model: Trained SVM model
        scaler: Feature scaler
        cnn_model: Trained CNN model
        X_features_test: Test features for SVM
        X_images_test: Test images for CNN
        y_test: Test labels
        svm_weight: Weight for SVM
        cnn_weight: Weight for CNN
        
    Returns:
        accuracy: Hybrid accuracy
    """
    print("\n" + "=" * 60)
    print("Evaluating Hybrid Model")
    print("=" * 60)
    
    # Get SVM predictions
    X_test_scaled = scaler.transform(X_features_test)
    svm_pred_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    # Get CNN predictions
    cnn_pred_prob = cnn_model.predict(X_images_test, verbose=0).flatten()
    
    # Combine predictions
    hybrid_prob = (svm_weight * svm_pred_prob) + (cnn_weight * cnn_pred_prob)
    hybrid_pred = (hybrid_prob > 0.5).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test, hybrid_pred)
    precision = precision_score(y_test, hybrid_pred)
    recall = recall_score(y_test, hybrid_pred)
    f1 = f1_score(y_test, hybrid_pred)
    
    print(f"\n✓ Hybrid Model Performance")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    # ROC-AUC for hybrid
    hybrid_roc_auc = roc_auc_score(y_test, hybrid_prob)
    print(f"  ROC-AUC: {hybrid_roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, hybrid_pred)
    print(cm)
    print(f"  True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    print("\nPer-Class Performance:")
    print(classification_report(y_test, hybrid_pred, target_names=['Normal', 'Abnormal']))
    
    return accuracy


def plot_training_history(history):
    """Plot CNN training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to {MODELS_DIR / 'training_history.png'}")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HeartGuard - Model Training")
    print("=" * 60)
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    X_features_train, X_images_train, y_train, _ = load_dataset('train')
    X_features_test, X_images_test, y_test, _ = load_dataset('test')
    
    # Train SVM
    svm_model, scaler = train_svm(X_features_train, y_train, X_features_test, y_test)
    
    # Save SVM model and scaler
    with open(MODELS_DIR / 'svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n✓ SVM model saved to {MODELS_DIR / 'svm_model.pkl'}")
    
    # Train CNN
    cnn_model, history = train_cnn(X_images_train, y_train, X_images_test, y_test)
    
    # Save CNN model
    cnn_model.save(MODELS_DIR / 'cnn_model.h5')
    print(f"\n✓ CNN model saved to {MODELS_DIR / 'cnn_model.h5'}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate hybrid model
    hybrid_acc = evaluate_hybrid(svm_model, scaler, cnn_model, X_features_test, X_images_test, y_test)
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    print(f"Models saved in: {MODELS_DIR}")
    print("\nNext steps:")
    print("1. Run the backend API: python backend/app.py")
    print("2. Open frontend/index.html in a browser")
    print("=" * 60)


if __name__ == "__main__":
    main()
