# HeartGuard - CHF Detection System

![HeartGuard](https://img.shields.io/badge/HeartGuard-CHF%20Detection-00d4ff)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-API-green)

A complete end-to-end web-based system for detecting Chronic Heart Failure (CHF) using heart sound (phonocardiogram) audio files. The system employs a hybrid approach combining traditional Machine Learning (SVM) and Deep Learning (CNN) for superior diagnostic performance.

## 🎯 Features

- **Hybrid ML/DL Approach**: Combines SVM and CNN predictions for maximum accuracy
- **Signal Processing**: Advanced audio preprocessing with 25-400Hz bandpass filtering and noise removal
- **Feature Extraction**: Comprehensive 90+ feature vector including MFCC, spectral, energy, and statistics
- **High Sensitivity**: Optimized for medical diagnosis with 99.34% recall for abnormal cases
- **Professional Dashboard**: Modern glassmorphism medical-style web interface with visualizations
- **Comprehensive Evaluation**: Detailed metrics for model validation

## � Key Metrics

- Accuracy: 99.33%
- Recall (Abnormal): 99%
- ROC-AUC: 1.0000

## �🚀 Model Performance (Verified)

| Model | Accuracy | Precision | Recall (Abnormal) | ROC-AUC |
|-------|----------|-----------|-------------------|---------|
| SVM | 99.33% | 0.99 | 0.99 | 1.0000 |
| CNN | 99.33% | 0.99 | 0.99 | 0.9996 |
| **Hybrid** | **99.33%**| **0.99** | **0.99** | **1.0000** |

## 📊 System Architecture

```
┌─────────────────┐
│   Audio Input   │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │  Preprocessing      │
    │  - Normalization    │
    │  - Bandpass Filter  │
    │  - Noise Removal    │
    └────┬────────────────┘
         │
    ┌────▼────────────────────────┐
    │                             │
┌───▼────────┐          ┌────────▼───┐
│ SVM Model  │          │ CNN Model  │
│ (Features) │          │ (Spectro)  │
└───┬────────┘          └────────┬───┘
    │                            │
    └────────┬───────────────────┘
             │
        ┌────▼──────────┐
        │ Hybrid Fusion │
        │ (Weighted Avg)│
        └────┬──────────┘
             │
        ┌────▼────────┐
        │  Prediction │
        │  + Confidence│
        └─────────────┘
```

## 🗂️ Project Structure

```
HeartGuard/
│
├── raw/                              # Raw PhysioNet dataset (Challenge 2016)
├── dataset/                          # Organized dataset (train/test splits)
├── backend/
│   ├── app.py                        # Flask API server
│   ├── train.py                      # Model training script
│   ├── models/                       # Saved weights & scalers
│   └── utils/
│       ├── preprocessing.py          # Signal processing pipeline
│       ├── features.py               # 90+ feature extraction logic
│       └── prediction.py             # Hybrid inference engine
├── frontend/
│   ├── index.html                    # Dashboard UI
│   ├── style.css                     # Medical-style glassmorphism
│   └── script.js                     # Async API logic
├── prepare_dataset.py                # Data ingestion & organization
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🛠️ Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Ensure the PhysioNet dataset is in `raw/` and run:
```bash
python prepare_dataset.py
```

### 3. Model Training

```bash
python backend/train.py
```

### 4. Start Application

```bash
python backend/app.py
```
Then open `frontend/index.html` in any modern web browser.

## 🔬 Technical Implementation

- **Sampling Rate**: 2000 Hz
- **Frequency Range**: 25-400 Hz (Optimized for Heart Sounds)
- **Feature Extraction**: MFCC, Zero-Crossing, Spectral Centroid, Bandwidth, Kurtosis, Skewness, etc.
- **CNN Architecture**: 3 Convolutional blocks with BatchNorm, Dropout, and Dense layers
- **Class Imbalance**: Handled via `balanced` class weights in both SVM and CNN
- **Threading**: Backend optimized for Windows thread-safety (Matplotlib 'Agg' backend)

---
**Disclaimer**: This system is for educational and research purposes only. It is NOT a medical device.
