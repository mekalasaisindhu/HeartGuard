"""
HeartGuard - Audio Preprocessing Utilities
Handles audio loading, filtering, noise removal, and spectrogram generation.
"""

import numpy as np
import librosa
import librosa.display
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for Flask thread safety
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Configuration
TARGET_SR = 2000  # Target sampling rate (Hz)
N_MELS = 128      # Number of Mel bands
HOP_LENGTH = 512  # Hop length for STFT


def load_audio(file_path, sr=TARGET_SR):
    """
    Load audio file, convert to mono, and normalize amplitude.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate
        
    Returns:
        audio: Normalized audio signal (mono, 1D array)
        sr: Sampling rate
    """
    # Load audio file, force mono conversion
    audio, original_sr = librosa.load(file_path, sr=sr, mono=True)
    
    # Ensure it's a 1D array (mono)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # Normalize amplitude to [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio, sr


def apply_bandpass_filter(audio, sr=TARGET_SR, lowcut=25, highcut=400):
    """
    Apply bandpass filter to isolate heart sound frequencies.
    
    Heart sounds typically occur in the 25-400 Hz range.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        
    Returns:
        filtered_audio: Bandpass filtered audio
    """
    # Design Butterworth bandpass filter
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are in valid range
    low = max(0.01, min(low, 0.99))
    high = max(0.01, min(high, 0.99))
    
    if low >= high:
        print(f"Warning: Invalid filter range, returning original audio")
        return audio
    
    # Create filter
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def remove_noise(audio, sr=TARGET_SR):
    """
    Apply noise reduction techniques.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        
    Returns:
        denoised_audio: Noise-reduced audio
    """
    # Apply median filtering to remove impulse noise
    from scipy.ndimage import median_filter
    
    # Convert to float for processing
    audio_denoised = median_filter(audio, size=3)
    
    return audio_denoised


def resample_audio(audio, original_sr, target_sr=TARGET_SR):
    """
    Resample audio to target sampling rate.
    
    Args:
        audio: Input audio signal
        original_sr: Original sampling rate
        target_sr: Target sampling rate
        
    Returns:
        resampled_audio: Resampled audio
    """
    if original_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    return audio


def generate_mel_spectrogram(audio, sr=TARGET_SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """
    Generate Mel spectrogram from audio signal.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        n_mels: Number of Mel bands
        hop_length: Hop length for STFT
        
    Returns:
        mel_spec: Mel spectrogram (dB scale)
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=2048
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def preprocess_audio(file_path):
    """
    Complete preprocessing pipeline for audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        dict containing:
            - original_audio: Original normalized audio
            - filtered_audio: Bandpass filtered audio
            - mel_spectrogram: Mel spectrogram
            - sr: Sampling rate
    """
    # Load audio
    audio, sr = load_audio(file_path)
    
    # Apply bandpass filter
    filtered_audio = apply_bandpass_filter(audio, sr)
    
    # Remove noise
    filtered_audio = remove_noise(filtered_audio, sr)
    
    # Generate Mel spectrogram
    mel_spec = generate_mel_spectrogram(filtered_audio, sr)
    
    return {
        'original_audio': audio,
        'filtered_audio': filtered_audio,
        'mel_spectrogram': mel_spec,
        'sr': sr,
        'anomalies': detect_anomalies(filtered_audio, sr)
    }


def detect_anomalies(audio, sr, threshold=2.5):
    """
    Detect anomalous spikes in heart sound energy.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        threshold: Z-score threshold for detection
        
    Returns:
        anomalies: List of dicts with 'time' and 'intensity'
    """
    # Calculate energy envelope (RMS)
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = int(0.01 * sr)    # 10ms hop
    
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Calculate energy variation (z-score)
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    if std_rms == 0: return []
    
    z_scores = (rms - mean_rms) / std_rms
    
    # Find peaks above threshold
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    anomalies = []
    # Group consecutive detections
    if len(anomaly_indices) > 0:
        current_group = [anomaly_indices[0]]
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] == anomaly_indices[i-1] + 1:
                current_group.append(anomaly_indices[i])
            else:
                # Add central point of the group
                mid_idx = current_group[len(current_group)//2]
                anomalies.append({
                    'time': float(times[mid_idx]),
                    'intensity': float(rms[mid_idx])
                })
                current_group = [anomaly_indices[i]]
        # Final group
        mid_idx = current_group[len(current_group)//2]
        anomalies.append({
            'time': float(times[mid_idx]),
            'intensity': float(rms[mid_idx])
        })
        
    return anomalies


def mel_spectrogram_to_image(mel_spec, target_size=(128, 128)):
    """
    Convert Mel spectrogram to image array for CNN input.
    
    Args:
        mel_spec: Mel spectrogram
        target_size: Target image size (height, width)
        
    Returns:
        image: Resized spectrogram image (normalized)
    """
    from skimage.transform import resize
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    
    # Resize to target size
    image = resize(mel_spec_norm, target_size, mode='reflect', anti_aliasing=True)
    
    # Add channel dimension for CNN (grayscale)
    image = np.expand_dims(image, axis=-1)
    
    return image


def plot_waveform(audio, sr, title="Waveform"):
    """
    Generate waveform plot as base64 encoded image.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        title: Plot title
        
    Returns:
        base64_image: Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 3))
    time = np.arange(len(audio)) / sr
    plt.plot(time, audio, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64


def plot_spectrogram(mel_spec, sr, title="Mel Spectrogram"):
    """
    Generate spectrogram plot as base64 encoded image.
    
    Args:
        mel_spec: Mel spectrogram
        sr: Sampling rate
        title: Plot title
        
    Returns:
        base64_image: Base64 encoded PNG image
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64
def estimate_heart_rate(audio, sr=TARGET_SR):
    """
    Robustly estimate heart rate (BPM) from a PCG signal using 
    autocorrelation of the Hilbert envelope.
    
    Args:
        audio: Preprocessed audio signal
        sr: Sampling rate
        
    Returns:
        bpm: Estimated heart rate
    """
    try:
        # 1. Compute the analytical signal and its envelope
        analytical_signal = signal.hilbert(audio)
        amplitude_envelope = np.abs(analytical_signal)
        
        # 2. Normalize and remove DC component from envelope
        envelope = amplitude_envelope - np.mean(amplitude_envelope)
        
        # 3. Compute autocorrelation of the envelope
        # We only need 'full' to find lags, then we'll slice it
        corr = signal.correlate(envelope, envelope, mode='full')
        lags = signal.correlation_lags(len(envelope), len(envelope), mode='full')
        
        # Slicing for positive lags only
        mid = len(lags) // 2
        corr = corr[mid:]
        lags = lags[mid:]
        
        # 4. Define acceptable BPM range (40 - 200 BPM)
        # Convert BPM to lag (number of samples between beats)
        # Lag = 60 * sr / BPM
        min_lag = int(60 * sr / 200) # Fast heart rate
        max_lag = int(60 * sr / 40)  # Slow heart rate
        
        if min_lag >= len(corr):
            return 0.0
            
        search_range = corr[min_lag:min(max_lag, len(corr))]
        if len(search_range) == 0:
            return 0.0
            
        # 5. Find strongest periodic component in range
        peak_lag_idx = np.argmax(search_range)
        peak_lag = min_lag + peak_lag_idx
        
        if peak_lag == 0:
            return 0.0
            
        bpm = (60 * sr) / peak_lag
        return float(bpm)
    except Exception as e:
        print(f"BPM Error: {e}")
        return 0.0
