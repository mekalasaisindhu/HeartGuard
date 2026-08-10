"""
HeartGuard - Hybrid Prediction Module
Combines SVM and CNN predictions for improved accuracy.
"""

import numpy as np
import pickle
from tensorflow import keras
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import preprocess_audio, mel_spectrogram_to_image, estimate_heart_rate
from utils.features import extract_all_features
import librosa


class HybridPredictor:
    """Hybrid predictor combining SVM and CNN models."""
    
    def __init__(self, svm_model_path, cnn_model_path, svm_weight=0.4, cnn_weight=0.6):
        """
        Initialize hybrid predictor.
        
        Args:
            svm_model_path: Path to trained SVM model
            cnn_model_path: Path to trained CNN model
            svm_weight: Weight for SVM prediction (default: 0.4)
            cnn_weight: Weight for CNN prediction (default: 0.6)
        """
        self.svm_weight = svm_weight
        self.cnn_weight = cnn_weight
        
        # Load models and scaler
        self.svm_model = self.load_svm_model(svm_model_path)
        self.cnn_model = self.load_cnn_model(cnn_model_path)
        
        # Load scaler
        scaler_path = Path(svm_model_path).parent / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler from {scaler_path}")
        else:
            print("Warning: Scaler not found. SVM predictions may be inaccurate.")
            self.scaler = None
        
    def load_svm_model(self, model_path):
        """Load trained SVM model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded SVM model from {model_path}")
        return model
    
    def load_cnn_model(self, model_path):
        """Load trained CNN model."""
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded CNN model from {model_path}")
        return model
    
    def predict_svm(self, audio, sr):
        """
        Get SVM prediction.
        
        Args:
            audio: Audio signal
            sr: Sampling rate
            
        Returns:
            prediction: 0 (normal) or 1 (abnormal)
            confidence: Prediction confidence
        """
        # Extract features
        features = extract_all_features(audio, sr)
        features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Get prediction result
        prediction = self.svm_model.predict(features)[0]
        
        # Get confidence (using predict_proba for calibrated scores)
        probs = self.svm_model.predict_proba(features)[0]
        confidence = float(probs[1]) if prediction == 1 else float(probs[0])
        
        return int(prediction), confidence
    
    def predict_cnn(self, mel_spec):
        """
        Get CNN prediction.
        
        Args:
            mel_spec: Mel spectrogram
            
        Returns:
            prediction: 0 (normal) or 1 (abnormal)
            confidence: Prediction confidence
        """
        # Convert spectrogram to image
        image = mel_spectrogram_to_image(mel_spec, target_size=(128, 128))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Get prediction
        prob = self.cnn_model.predict(image, verbose=0)[0][0]
        
        # Convert probability to prediction
        prediction = 1 if prob > 0.5 else 0
        
        # Confidence is the probability
        confidence = float(prob) if prediction == 1 else float(1 - prob)
        
        return int(prediction), float(confidence)
    
    def hybrid_predict(self, audio_file_path, preprocessed=None):
        """
        Perform hybrid prediction combining SVM and CNN.
        
        Args:
            audio_file_path: Path to audio file
            preprocessed: Optional preprocessed audio dict to avoid duplicate processing
            
        Returns:
            dict containing:
                - prediction: 'Normal' or 'Abnormal'
                - confidence: Overall confidence score
                - svm_prediction: SVM prediction
                - svm_confidence: SVM confidence
                - cnn_prediction: CNN prediction
                - cnn_confidence: CNN confidence
        """
        print("[PREDICT] Starting hybrid_predict")
        if preprocessed is None:
            print("[PREDICT] No preprocessed audio provided, calling preprocess_audio")
            preprocess_start = time.time()
            preprocessed = preprocess_audio(audio_file_path)
            print(f"[PREDICT] Internal preprocess_audio completed in {time.time() - preprocess_start:.2f} seconds")
        else:
            print("[PREDICT] Using preprocessed audio from calling endpoint")

        audio = preprocessed['filtered_audio']
        mel_spec = preprocessed['mel_spectrogram']
        sr = preprocessed['sr']
        
        # Get SVM prediction
        svm_start = time.time()
        print("[PREDICT] Starting SVM prediction")
        svm_pred, svm_conf = self.predict_svm(audio, sr)
        print(f"[PREDICT] SVM prediction completed in {time.time() - svm_start:.2f} seconds")
        
        # Get CNN prediction
        cnn_start = time.time()
        print("[PREDICT] Starting CNN prediction")
        cnn_pred, cnn_conf = self.predict_cnn(mel_spec)
        print(f"[PREDICT] CNN prediction completed in {time.time() - cnn_start:.2f} seconds")
        
        # Combine predictions using weighted average
        # Convert predictions to probabilities
        svm_prob = svm_conf if svm_pred == 1 else (1 - svm_conf)
        cnn_prob = cnn_conf if cnn_pred == 1 else (1 - cnn_conf)
        
        # Weighted average
        combined_prob = (self.svm_weight * svm_prob) + (self.cnn_weight * cnn_prob)
        
        # Final prediction
        final_prediction = 1 if combined_prob > 0.5 else 0
        final_confidence = combined_prob if final_prediction == 1 else (1 - combined_prob)
        
        # Convert to labels
        prediction_label = 'Abnormal' if final_prediction == 1 else 'Normal'
        svm_label = 'Abnormal' if svm_pred == 1 else 'Normal'
        cnn_label = 'Abnormal' if cnn_pred == 1 else 'Normal'
        
        # Extract BPM
        try:
            bpm = estimate_heart_rate(audio, sr=sr)
        except:
            bpm = 0.0

        # Generate XAI Insights
        insights = []
        
        # 1. Model Agreement
        if svm_label == cnn_label:
            insights.append(f"Both AI models ({svm_label}) are in agreement.")
        else:
            insights.append("Models show slight disagreement; caution advised.")
            
        # 2. Confidence Insight
        if final_confidence > 0.85:
            insights.append("High model confidence for this analysis.")
        elif final_confidence < 0.70:
            insights.append("Lower confidence; suggest re-recording or noise check.")
            
        # 3. Heart Rate Insight
        if bpm > 100:
            insights.append(f"Tachycardia detected (~{round(bpm)} BPM).")
        elif 0 < bpm < 60:
            insights.append(f"Bradycardia detected (~{round(bpm)} BPM).")
            
        # 4. Anomaly Insight
        anomalies = preprocessed.get('anomalies', [])
        if len(anomalies) > 0:
            insights.append(f"Identified {len(anomalies)} energetic anomalies/murmur points.")

        return {
            'prediction': prediction_label,
            'confidence': float(final_confidence),
            'svm_prediction': svm_label,
            'svm_confidence': float(svm_conf),
            'cnn_prediction': cnn_label,
            'cnn_confidence': float(cnn_conf),
            'bpm': bpm,
            'insights': insights
        }


def load_models(models_dir='backend/models'):
    """
    Load trained models and create hybrid predictor.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        HybridPredictor instance
    """
    models_path = Path(models_dir)
    svm_path = models_path / 'svm_model.pkl'
    cnn_path = models_path / 'cnn_model.h5'
    
    if not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found: {svm_path}")
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path}")
    
    predictor = HybridPredictor(str(svm_path), str(cnn_path))
    return predictor


if __name__ == "__main__":
    # Test prediction
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prediction.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Load models
    predictor = load_models()
    
    # Make prediction
    result = predictor.hybrid_predict(audio_file)
    
    print("\n" + "=" * 50)
    print("HeartGuard - Prediction Result")
    print("=" * 50)
    print(f"Final Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print()
    print(f"SVM: {result['svm_prediction']} ({result['svm_confidence']:.2%})")
    print(f"CNN: {result['cnn_prediction']} ({result['cnn_confidence']:.2%})")
    print("=" * 50)
