import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchaudio.functional import resample

# Load HuBERT model
discrete_units = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).to(torch.device("cuda"))
discrete_units.eval()

def get_pitch_contour(file_path, sr):
    """Extract feature from audio file."""
    x, sr = librosa.load(file_path, sr=sr, mono=True)
    x = torch.tensor(x).unsqueeze(0).to(torch.device('cuda'))

    #zero-pad end if x is smaller than input to network
    if x.shape[-1] < 65536:
        zeros = torch.zeros(1, 65536 - x.shape[-1]).to(torch.device('cuda'))
        x = torch.cat((x, zeros), dim=-1)

    #zero pad end with one second to ensure that the offset does not go out of range
    zeros = torch.zeros(1, sr).to(torch.device('cuda'))
    x = torch.cat((x, zeros), dim=-1)
    
    x_resampled = resample(x, sr, 16000)
    units = discrete_units.units(x_resampled.unsqueeze(0))
    
    return units.detach().cpu()

def process_audio_directory(base_dirs, output_path, sample_rate):
    """
    Process multiple directories of audio files.
    
    Args:
        base_dirs (str or list): Single directory path or list of directory paths
        output_path (str): Path to save the output pickle file
        sample_rate (int): Target sample rate for audio processing
    """
    # Convert single directory to list for consistent handling
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    
    audio_data = {}
    
    # Process each base directory
    for base_dir in base_dirs:
        print(f"\nProcessing directory: {base_dir}")
        
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    file_path = os.path.join(root, file)
                    print(f"Processing {file_path}...")
                    
                    try:
                        units = get_pitch_contour(file_path, sample_rate)
                        audio_data[file_path] = {
                            'units': units
                        }
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        continue
    
    # Save all processed data
    with open(output_path, 'wb') as f:
        pickle.dump(audio_data, f)
    print(f"\nSaved units to {output_path}")
    print(f"Processed {len(audio_data)} files in total")

# Example usage
base_directories = [
    "vctk-small",
    "val-set-test",
]
sample_rate = 44100
output_file = "metadata.pkl"

process_audio_directory(base_directories, output_file, sample_rate)