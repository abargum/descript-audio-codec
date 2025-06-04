import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from create_kmeans import kmeans
from torchaudio.functional import resample
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, AutoModel, HubertConfig

# Load HuBERT model
discrete_units = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).to(torch.device("cuda"))
discrete_units.eval()

# Load multi HuBERT model with pre-trained kmeans
pretrained_path = "scripts/utils/kmeans_200_multi.pt"
config = HubertConfig.from_pretrained("utter-project/mHuBERT-147", output_hidden_states=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
discrete_units_multi = AutoModel.from_pretrained("utter-project/mHuBERT-147", config=config).to(torch.device("cuda"))
discrete_units_multi.eval()
kmean_multi = kmeans(pretrained=True, clusters=200, checkpoint=pretrained_path)

def get_hubert_units(audio):
    units = discrete_units.units(audio.unsqueeze(0))
    return units.detach().cpu()

def get_hubert_units_multi(audio):
    output = discrete_units_multi(audio)
    output = output.hidden_states[6].squeeze(0)
    units = kmean_multi.predict(output.squeeze().detach().cpu().numpy())
    units = torch.tensor(units, dtype=torch.long)
    return units.detach().cpu()

def get_features(file_path, sr):
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

    hubert_units = get_hubert_units(x_resampled)
    hubert_units_multi = get_hubert_units_multi(x_resampled)
        
    return hubert_units, hubert_units_multi

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
                        hubert_units, hubert_units_multi = get_features(file_path, sample_rate)
                        
                        audio_data[file_path] = {
                            'hubert_units': hubert_units,
                            'hubert_units_multi': hubert_units_multi
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
    "validation-set",
]
sample_rate = 44100
output_file = "metadata_w_multi.pkl"

process_audio_directory(base_directories, output_file, sample_rate)