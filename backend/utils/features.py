"""
HeartGuard - Feature Extraction Utilities
Extracts features from audio signals for traditional ML models (SVM).
"""

import numpy as np
import librosa
from scipy import stats


def extract_mfcc(audio, sr, n_mfcc=13):
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) features.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        mfcc_features: Statistical features from MFCCs (mean, std, min, max)
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Extract statistical features from each MFCC coefficient
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    
    # Concatenate all features
    mfcc_features = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])
    
    return mfcc_features.astype(np.float64)


def extract_spectral_features(audio, sr):
    """
    Extract spectral features.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        
    Returns:
        spectral_features: Array of spectral features
    """
    features = []
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([
        np.mean(zcr),
        np.std(zcr),
        np.min(zcr),
        np.max(zcr)
    ])
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.extend([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.min(spectral_centroid),
        np.max(spectral_centroid)
    ])
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.extend([
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.min(spectral_bandwidth),
        np.max(spectral_bandwidth)
    ])
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.extend([
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.min(spectral_rolloff),
        np.max(spectral_rolloff)
    ])
    
    return np.array(features, dtype=np.float64)


def extract_energy_features(audio):
    """
    Extract energy-based features.
    
    Args:
        audio: Audio signal
        
    Returns:
        energy_features: Array of energy features
    """
    features = []
    
    # RMS energy
    rms = librosa.feature.rms(y=audio)
    features.extend([
        np.mean(rms),
        np.std(rms),
        np.min(rms),
        np.max(rms)
    ])
    
    # Total energy
    total_energy = np.sum(audio ** 2)
    features.append(total_energy)
    
    return np.array(features, dtype=np.float64)


def extract_statistical_features(audio):
    """
    Extract statistical features from raw audio signal.
    
    Args:
        audio: Audio signal
        
    Returns:
        statistical_features: Array of statistical features
    """
    features = []
    
    # Basic statistics
    features.append(np.mean(audio))
    features.append(np.std(audio))
    features.append(np.min(audio))
    features.append(np.max(audio))
    features.append(np.median(audio))
    
    # Higher-order statistics
    features.append(stats.skew(audio))  # Skewness
    features.append(stats.kurtosis(audio))  # Kurtosis
    
    # Percentiles
    features.append(np.percentile(audio, 25))
    features.append(np.percentile(audio, 75))
    
    return np.array(features, dtype=np.float64)


def extract_temporal_features(audio, sr):
    """
    Extract temporal features.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        
    Returns:
        temporal_features: Array of temporal features
    """
    features = []
    
    # Duration
    duration = len(audio) / sr
    features.append(duration)
    
    # Tempo (beat tracking)
    try:
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # Ensure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)
        features.append(tempo)
    except:
        features.append(0.0)
    
    return np.array(features, dtype=np.float64)


def extract_all_features(audio, sr):
    """
    Extract all features from audio signal.
    
    This combines MFCC, spectral, energy, statistical, and temporal features
    into a single feature vector for ML model input.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        
    Returns:
        feature_vector: Complete feature vector
    """
    # Extract all feature types
    mfcc_features = extract_mfcc(audio, sr)
    spectral_features = extract_spectral_features(audio, sr)
    energy_features = extract_energy_features(audio)
    statistical_features = extract_statistical_features(audio)
    temporal_features = extract_temporal_features(audio, sr)
    
    # Concatenate all features
    feature_vector = np.concatenate([
        mfcc_features,
        spectral_features,
        energy_features,
        statistical_features,
        temporal_features
    ])
    
    return feature_vector


def extract_features_from_file(file_path, sr=2000):
    """
    Extract features directly from audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate
        
    Returns:
        feature_vector: Complete feature vector
    """
    # Load audio
    audio, sr = librosa.load(file_path, sr=sr)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Extract features
    features = extract_all_features(audio, sr)
    
    return features
