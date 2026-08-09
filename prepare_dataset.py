"""
HeartGuard - Dataset Preparation Script
Organizes PhysioNet Heart Sound dataset into train/test splits with normal/abnormal labels.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

# Configuration
RAW_DIR = Path("raw")
DATASET_DIR = Path("dataset")
ANNOTATIONS_FILE = RAW_DIR / "annotations" / "Online_Appendix_training_set.csv"

# Training folders (will go to train split)
TRAINING_FOLDERS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]

# Validation folder (will go to test split)
VALIDATION_FOLDER = "validation"


def create_directory_structure():
    """Create the dataset directory structure."""
    print("Creating directory structure...")
    
    # Create main directories
    (DATASET_DIR / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "train" / "abnormal").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "test" / "abnormal").mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")


def load_annotations():
    """Load and parse the annotations CSV file."""
    print(f"Loading annotations from {ANNOTATIONS_FILE}...")
    
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")
    
    # Load CSV
    df = pd.read_csv(ANNOTATIONS_FILE)
    
    # Create a dictionary mapping recording name to label
    # Class: -1 = Normal, 1 = Abnormal
    annotations = {}
    for _, row in df.iterrows():
        # The CSV uses 'Challenge record name' instead of 'Recording'
        recording = str(row['Challenge record name'])
        # The CSV uses 'Class (-1=normal 1=abnormal)' instead of 'Class'
        label = row['Class (-1=normal 1=abnormal)']
        
        # Map -1 to 'normal' and 1 to 'abnormal'
        if label == -1:
            annotations[recording] = 'normal'
        elif label == 1:
            annotations[recording] = 'abnormal'
        else:
            print(f"Warning: Unknown label {label} for recording {recording}")
    
    print(f"✓ Loaded {len(annotations)} annotations")
    return annotations


def copy_files(source_folders, target_split, annotations):
    """
    Copy audio files from source folders to target split directory.
    
    Args:
        source_folders: List of folder names to copy from
        target_split: 'train' or 'test'
        annotations: Dictionary mapping recording names to labels
    """
    stats = {'normal': 0, 'abnormal': 0, 'skipped': 0}
    
    for folder in source_folders:
        folder_path = RAW_DIR / folder
        
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        print(f"Processing {folder}...")
        
        # Get all .wav files in the folder
        wav_files = list(folder_path.glob("*.wav"))
        
        for wav_file in wav_files:
            # Get recording name (without .wav extension)
            recording_name = wav_file.stem
            
            # Get label from annotations
            if recording_name not in annotations:
                print(f"  Warning: No annotation for {recording_name}, skipping...")
                stats['skipped'] += 1
                continue
            
            label = annotations[recording_name]
            
            # Determine target directory
            target_dir = DATASET_DIR / target_split / label
            target_file = target_dir / wav_file.name
            
            # Copy file
            shutil.copy2(wav_file, target_file)
            stats[label] += 1
        
        print(f"  ✓ Processed {len(wav_files)} files from {folder}")
    
    return stats


def main():
    """Main execution function."""
    print("=" * 60)
    print("HeartGuard - Dataset Preparation")
    print("=" * 60)
    print()
    
    # Step 1: Create directory structure
    create_directory_structure()
    print()
    
    # Step 2: Load annotations
    annotations = load_annotations()
    print()
    
    # Step 3: Copy training files
    print("Copying training files...")
    train_stats = copy_files(TRAINING_FOLDERS, 'train', annotations)
    print(f"✓ Training set: {train_stats['normal']} normal, {train_stats['abnormal']} abnormal, {train_stats['skipped']} skipped")
    print()
    
    # Step 4: Copy validation files (test set)
    print("Copying validation files (test set)...")
    test_stats = copy_files([VALIDATION_FOLDER], 'test', annotations)
    print(f"✓ Test set: {test_stats['normal']} normal, {test_stats['abnormal']} abnormal, {test_stats['skipped']} skipped")
    print()
    
    # Summary
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Train set: {train_stats['normal'] + train_stats['abnormal']} files")
    print(f"  - Normal: {train_stats['normal']}")
    print(f"  - Abnormal: {train_stats['abnormal']}")
    print()
    print(f"Test set: {test_stats['normal'] + test_stats['abnormal']} files")
    print(f"  - Normal: {test_stats['normal']}")
    print(f"  - Abnormal: {test_stats['abnormal']}")
    print()
    print("Dataset structure:")
    print("dataset/")
    print("├── train/")
    print("│   ├── normal/")
    print("│   └── abnormal/")
    print("└── test/")
    print("    ├── normal/")
    print("    └── abnormal/")
    print()


if __name__ == "__main__":
    main()
