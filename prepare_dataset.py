# Steps:
# 1) Load filelist of audio that needs to be extracted
# 2) For each audio file, extract features with padding, then remove padding from features
# 3) Save features to CSV file

import numpy as np
import pandas as pd
import os
import feature_extractor as fe
import feature_selection as fs

feature_extractor = fe.FeatureExtractor('hubert_l')

def load_filelist(file_path):
    """
    Loads a list of filenames from a text file.
    
    Args:
        file_path: Path to the text file containing the list of filenames.
    Returns:
        A list of filenames.
    """
    if not os.path.exists(file_path):
        print(f"Error: File list not found at {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        filelist_items = [line.strip() for line in f if line.strip()]
    
    return filelist_items

def extract_and_save_features(filelist, data_directory, feature_output_dir):
    """
    Extracts features from audio files and saves them as .npy files.
    
    Args:
        filelist: A list of filenames to process.
        data_directory: Directory where the audio files are located.
        feature_output_dir: Directory where the extracted features will be saved.
    """

    for filename in filelist:
        # Only process if features do not already exist
        if not os.path.exists(os.path.join(feature_output_dir, filename.replace('.wav', '_features.npy'))):
            file_path = os.path.join(data_directory, filename)
            print(f"Processing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue

            # Extract features
            features = feature_extractor.get_24th_layer_features_averages(
                file_path, extract_with_padding=True
            ).flatten()
            np.save(os.path.join(feature_output_dir, filename.replace('.wav', '_features.npy')), features)
