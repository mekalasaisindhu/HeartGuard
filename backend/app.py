"""
HeartGuard - Flask Backend API
Provides endpoints for audio upload, processing, and CHF prediction.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os
import tempfile
import traceback
import sys
import csv
from datetime import datetime
import numpy as np
import librosa

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.preprocessing import preprocess_audio, plot_waveform, plot_spectrogram, HOP_LENGTH
from utils.prediction import load_models

# Initialize Flask app with frontend static folder
app = Flask(
    __name__,
    static_folder=str(Path(__file__).resolve().parent.parent / 'frontend'),
    static_url_path=''
)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'heartguard_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

# Load models at startup
print("Loading models...")
try:
    predictor = load_models('backend/models')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Please train models first using: python backend/train.py")
    predictor = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload audio file endpoint.
    
    Returns:
        JSON with file_path and status
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Save file
        filename = f"upload_{os.urandom(8).hex()}_{file.filename}"
        file_path = UPLOAD_FOLDER / filename
        file.save(str(file_path))
        
        return jsonify({
            'status': 'success',
            'file_path': str(file_path),
            'filename': file.filename
        })
    
    except Exception as e:
        print(f"Error in upload: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_audio():
    """
    Process audio file and return visualizations.
    
    Expects JSON with 'file_path' field.
    
    Returns:
        JSON with waveform and spectrogram images (base64)
    """
    try:
        # Get file path from request
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Preprocess audio
        preprocessed = preprocess_audio(file_path)
        
        # Generate visualizations
        original_waveform = plot_waveform(
            preprocessed['original_audio'],
            preprocessed['sr'],
            title='Original Waveform'
        )
        
        filtered_waveform = plot_waveform(
            preprocessed['filtered_audio'],
            preprocessed['sr'],
            title='Filtered Waveform'
        )
        
        spectrogram = plot_spectrogram(
            preprocessed['mel_spectrogram'],
            preprocessed['sr'],
            title='Mel Spectrogram'
        )
        
        return jsonify({
            'status': 'success',
            'original_waveform': original_waveform,
            'filtered_waveform': filtered_waveform,
            'spectrogram': spectrogram
        })
    
    except Exception as e:
        print(f"Error in process: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict CHF from audio file.
    
    Expects JSON with 'file_path' field.
    
    Returns:
        JSON with prediction result and confidence
    """
    try:
        # Check if models are loaded
        if predictor is None:
            return jsonify({
                'error': 'Models not loaded. Please train models first using: python backend/train.py'
            }), 503
        
        # Get file path from request
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Make prediction
        result = predictor.hybrid_predict(file_path)
        
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'svm_prediction': result['svm_prediction'],
            'svm_confidence': result['svm_confidence'],
            'cnn_prediction': result['cnn_prediction'],
            'cnn_confidence': result['cnn_confidence'],
            'bpm': result.get('bpm', 0),
            'insights': result.get('insights', [])
        })
    
    except Exception as e:
        print(f"Error in predict: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Complete analysis: process and predict in one call with interactive data.
    """
    try:
        # Get file path from request
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not Path(file_path).exists():
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Process audio
        preprocessed = preprocess_audio(file_path)
        sr = preprocessed['sr']
        
        # Prepare Plotly Data
        # 1. Waveform (Downsampled to 2000 points max)
        audio = preprocessed['filtered_audio']
        max_pts = 2000
        step = max(1, len(audio) // max_pts)
        waveform_data = audio[::step].tolist()
        waveform_times = (np.arange(0, len(audio), step) / sr).tolist()

        # 2. Spectrogram (Downsampled for Plotly Heatmap)
        # Mel spec is (n_mels, n_frames)
        mel_spec = preprocessed['mel_spectrogram']
        # Mel scales for Y-axis (Frequencies)
        frequencies = librosa.mel_frequencies(n_mels=mel_spec.shape[0], fmin=0, fmax=sr/2).tolist()
        # Time scales for X-axis
        # HOP_LENGTH is defined in preprocessing.py as 512
        spec_times = librosa.frames_to_time(np.arange(mel_spec.shape[1]), sr=sr, hop_length=HOP_LENGTH).tolist()
        
        # Downsample spectrogram slightly if too large (e.g. max 500 frames for UI performance)
        frame_step = max(1, mel_spec.shape[1] // 500)
        plotly_spec = mel_spec[:, ::frame_step].tolist()
        plotly_spec_times = spec_times[::frame_step]

        # Generate static visualizations as fallback
        original_waveform_img = plot_waveform(
            preprocessed['original_audio'],
            sr,
            title='Original Waveform'
        )
        
        filtered_waveform_img = plot_waveform(
            preprocessed['filtered_audio'],
            sr,
            title='Filtered Waveform'
        )
        
        spectrogram_img = plot_spectrogram(
            mel_spec,
            sr,
            title='Mel Spectrogram'
        )
        
        # Make prediction
        prediction_result = None
        if predictor is not None:
            prediction_result = predictor.hybrid_predict(file_path)
        
        response = {
            'status': 'success',
            'visualizations': {
                'original_waveform': original_waveform_img,
                'filtered_waveform': filtered_waveform_img,
                'spectrogram': spectrogram_img
            },
            'plotly_data': {
                'waveform': {
                    'y': waveform_data,
                    'x': waveform_times
                },
                'spectrogram': {
                    'z': plotly_spec,
                    'x': plotly_spec_times,
                    'y': frequencies
                },
                'anomalies': preprocessed['anomalies']
            }
        }
        
        if prediction_result:
            response['prediction'] = {
                'result': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'svm_prediction': prediction_result['svm_prediction'],
                'svm_confidence': prediction_result['svm_confidence'],
                'cnn_prediction': prediction_result['cnn_prediction'],
                'cnn_confidence': prediction_result['cnn_confidence'],
                'bpm': prediction_result.get('bpm', 0),
                'insights': prediction_result.get('insights', [])
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Log user feedback for future model training.
    """
    try:
        data = request.get_json()
        filename = data.get('filename', 'unknown')
        prediction = data.get('prediction', 'unknown')
        user_label = data.get('user_label', 'unknown')
        is_correct = data.get('is_correct', False)
        
        feedback_file = Path('backend/feedback_log.csv')
        file_exists = feedback_file.exists()
        
        with open(feedback_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'filename', 'model_prediction', 'user_label', 'is_correct'])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                filename,
                prediction,
                user_label,
                is_correct
            ])
            
        return jsonify({'status': 'success', 'message': 'Feedback received'})
    
    except Exception as e:
        print(f"Error in feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the frontend single-page app and static assets."""
    if path != '' and Path(app.static_folder, path).exists():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'

    print("\n" + "=" * 60)
    print("HeartGuard Backend API")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:{port}")
    print("\nAvailable endpoints:")
    print("  GET  /health    - Health check")
    print("  POST /upload    - Upload audio file")
    print("  POST /process   - Process audio and get visualizations")
    print("  POST /predict   - Get CHF prediction")
    print("  POST /analyze   - Complete analysis (process + predict)")
    print("=" * 60)
    print()
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, use_reloader=False)
