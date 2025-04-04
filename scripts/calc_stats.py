import os
import torch
import librosa
import pickle
import torch.nn as nn
import numpy as np
from torchfcpe import spawn_bundled_infer_model
from modules.encoder import SpeakerEncoder
from modules.pqmf import CachedPQMF as PQMF

pitch_model = spawn_bundled_infer_model(device="cuda:0")
speaker_encoder = SpeakerEncoder()
pqmf = PQMF(attenuation=100, n_band=16)

def load_speaker_statedict(path):
    loaded_state = torch.load(path, map_location="cuda")
        
    newdict = {}
    pqmfdict = {}
    delete_list = []
        
    for name, param in loaded_state.items():
        new_name = name.replace("__S__.", "")
            
        if "pqmf" in new_name:
            new_name = new_name.replace("pqmf.", "")
            pqmfdict[new_name] = param
        else:
            newdict[new_name] = param
                
        delete_list.append(name)
    loaded_state.update(newdict)
    for name in delete_list:
        del loaded_state[name]
                
    return loaded_state, pqmfdict

# Load state dictionary
spk_state, pqmf_state = load_speaker_statedict("scripts/utils/model000000075.model")
speaker_encoder.load_state_dict(spk_state)
speaker_encoder.eval()

def adjust_audio_length(y):
    audio_length = y.shape[-1]
    if audio_length > 131072:
        y = y[:, :, :131072]
    else:
        zero_length = 131072 - audio_length
        zeros = torch.zeros(1, 1, zero_length)
        y = torch.cat((y, zeros), dim=-1)
    return y

def calculate_embedding(audio):
    audio = adjust_audio_length(audio)
    audio_multiband = pqmf(audio)
    emb = speaker_encoder(audio_multiband)
    return emb

def calculate_speaker_stats(folder_path, sample_rate=44100):
    all_f0_values = []
    all_embeddings = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.wav', '.mp3', '.flac')):
            audio, sr = librosa.load(file_path, sr=sample_rate)
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            
            f0 = pitch_model.infer(audio_tensor, sr=sr, decoder_mode='local_argmax', threshold=0.006, f0_min=50, f0_max=550)
            f0_nonzero = f0[f0 > 0]
            all_f0_values.append(f0_nonzero)
            
            # Note: Audio needs to be expanded to 3D for embedding: [batch, channels, time]
            audio_tensor_3d = audio_tensor.unsqueeze(0)
            emb = calculate_embedding(audio_tensor_3d)
            all_embeddings.append(emb)
    
    # Concatenate all F0 values and calculate global stats
    if all_f0_values:
        all_f0_values = torch.cat(all_f0_values)
        global_mean = torch.mean(all_f0_values).item()
        global_std = torch.std(all_f0_values).item()
    else:
        global_mean = 0.0
        global_std = 0.0
    
    # Average all embeddings
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        global_emb = torch.mean(all_embeddings, dim=0).detach().cpu().numpy()
    else:
        global_emb = torch.zeros(speaker_encoder.embedding_dim).numpy()
    
    return global_mean, global_std, global_emb

def calculate_all_speakers_stats(root_folder, sample_rate=44100):
    speaker_stats = {}
    for speaker_folder in os.listdir(root_folder):
        speaker_path = os.path.join(root_folder, speaker_folder)
        if os.path.isdir(speaker_path):  # Ensure it's a folder
            print(f"Processing speaker: {speaker_folder}")
            mean_f0, std_f0, emb = calculate_speaker_stats(speaker_path, sample_rate)
            
            # Structure the stats dictionary in the desired format
            speaker_stats[speaker_folder] = {
                'avg_emb': emb,
                'f0_mean': mean_f0,
                'f0_std': std_f0
            }
    
    return speaker_stats

# Example usage
root_folder = "../libri-speakers"
all_speaker_stats = calculate_all_speakers_stats(root_folder)

# Print results
for speaker, stats in all_speaker_stats.items():
    print(f"Speaker: {speaker}, Mean F0: {stats['f0_mean']:.2f} Hz, Std F0: {stats['f0_std']:.2f} Hz")

# Save as pickle file
file_path = 'scripts/utils/speaker_emb_dict_libri.pkl'
os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
with open(file_path, 'wb') as file:
    pickle.dump(all_speaker_stats, file)

print(f"Speaker stats saved to {file_path}")