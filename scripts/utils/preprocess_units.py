import os
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from create_kmeans import kmeans
from torchaudio.functional import resample
from transformers import AutoProcessor, WavLMModel
from transformers import Wav2Vec2FeatureExtractor, AutoModel, HubertConfig

# Load multi-speaker HuBERT model
pretrained_path = "scripts/utils/kmeans_200_multi.pt"
config = HubertConfig.from_pretrained("utter-project/mHuBERT-147", output_hidden_states=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
model_hubert = AutoModel.from_pretrained("utter-project/mHuBERT-147", config=config).to(torch.device("cuda"))
model_hubert.eval()
kmean_hubert = kmeans(pretrained=True, clusters=200, checkpoint=pretrained_path)

# Load WavLM model with pre-trained kmeans
pretrained_path = "scripts/utils/kmeans_512_wavlm.pt"
num_clusters = 512
model_wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus").to(torch.device("cuda"))
model_wavlm.eval()
kmean_unit_extractor = kmeans(pretrained=True, clusters=num_clusters, checkpoint=pretrained_path)

#model_name = "facebook/w2v-bert-2.0"  # Use the appropriate W2VBert model
#processor = AutoProcessor.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name).to(device)

def get_wavlm_units(audio):
    output = model_wavlm(audio)
    output = output.last_hidden_state.squeeze(0)
    units = kmean_unit_extractor.predict(output.squeeze().detach().cpu().numpy())
    units = torch.tensor(units, dtype=torch.long)
    return units, output.squeeze().detach().cpu()

def get_hubert_units(audio):
    output = model_hubert(audio)
    output = output.hidden_states[6].squeeze(0)
    units = kmean_hubert.predict(output.squeeze().detach().cpu().numpy())
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
    wavlm_units, wavlm_output = get_wavlm_units(x_resampled)

    #if wavlm units a smaller repeat last value
    if wavlm_units.shape[0] < hubert_units.shape[0]:
        diff = hubert_units.shape[0] - wavlm_units.shape[0]
        last_val = wavlm_units[-1]
        wavlm_units = torch.cat([wavlm_units, last_val.repeat(diff)])
        
        last_val = wavlm_output[-1:, :]  
        wavlm_output = torch.cat([wavlm_output, last_val], dim=0)
        
    return hubert_units, wavlm_units, wavlm_output

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
                        hubert_units, wavlm_units, wavlm_output = get_features(file_path, sample_rate)
                        
                        audio_data[file_path] = {
                            'hubert_units': hubert_units,
                            'wavlm_units': wavlm_units,
                            'wavlm_output': wavlm_output
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
    "VCTK-Corpus/wav48",
    "validation-set",
]
sample_rate = 44100
output_file = "metadata_w_wavlm_full.pkl"

process_audio_directory(base_directories, output_file, sample_rate)