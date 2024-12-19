import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchaudio.functional import resample

discrete_units = torch.hub.load("bshall/hubert:main",f"hubert_discrete",
                                trust_repo=True).to(torch.device("cuda"))

# Define a function to extract feature
def get_pitch_contour(file_path, sr):
    x, sr = librosa.load(file_path, sr=sr) 
    x = torch.tensor(x[:16384]).unsqueeze(0).to(torch.device('cuda'))
    
    if x.shape[-1] < 16384:
        padding = torch.zeros(1, 16384 - x.shape[-1]).to(torch.device('cuda'))
        x = torch.cat([x, padding], dim=-1)
        
    x_resampled = resample(x, sample_rate, 16000)
    units = discrete_units.units(x_resampled.unsqueeze(0))
    
    return units.detach().cpu()

# Traverse the directory and process audio files
def process_audio_directory(base_dir, output_path, sample_rate):
    audio_data = {}
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                
                units = get_pitch_contour(file_path, sample_rate)
                audio_data[file_path] = {
                    'units': units
                }

    with open(output_path, 'wb') as f:
        pickle.dump(audio_data, f)
    print(f"Saved pitch contours to {output_path}")

# Specify the directory and output file
base_directory = "../vctk-small"
sample_rate = 44100
output_file = "metadata.pkl"
process_audio_directory(base_directory, output_file, sample_rate)
