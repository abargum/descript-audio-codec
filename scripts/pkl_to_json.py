import os
import pickle
import json
import numpy as np
import torch
import random
import librosa
from rave.blocks2 import SpeakerRAVE, EncoderV2
from rave.pqmf import CachedPQMF as PQMF

def remove_silence(audio, sr, min_silence_duration=0.5, threshold_db=-40):
    threshold_amp = librosa.db_to_amplitude(threshold_db)
    
    frame_length = 1024
    hop_length = 512
    
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    speech_frames = energy > threshold_amp
    min_silence_frames = int(min_silence_duration * sr / hop_length)
    speech_samples = np.zeros_like(audio, dtype=bool)
    
    for i in range(len(speech_frames)):
        start_sample = i * hop_length
        end_sample = min(start_sample + hop_length, len(audio))
        speech_samples[start_sample:end_sample] = speech_frames[i]
    
    extend_samples = int(0.1 * sr)  # 100ms extension on each side
    speech_samples = np.convolve(speech_samples, np.ones(extend_samples), mode='same') > 0
    non_silent_indices = np.where(speech_samples)[0]
    
    if len(non_silent_indices) == 0:
        print("Warning: No speech detected in audio")
        return audio  # Return original if no speech detected
    
    filtered_audio = audio[non_silent_indices]
    
    print(f"Removed silence: original length {len(audio)/sr:.2f}s, new length {len(filtered_audio)/sr:.2f}s")
    return filtered_audio

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

if __name__ == "__main__":

    pqmf = PQMF(attenuation = 100, n_band = 16)
    speaker_encoder = SpeakerRAVE()
    spk_state, pqmf_state = load_speaker_statedict("scripts/rave/model000000075.model")
    speaker_encoder.load_state_dict(spk_state)
    speaker_encoder.eval()
    
    folder_path = "VCTK-Corpus/wav48"

    file_path = 'scripts/utils/speaker_emb_dict.pkl'
    with open(file_path, 'rb') as file:
        speaker_dict = pickle.load(file)

    # Convert NumPy arrays to lists
    json_ready_data = {}
    for speaker, features in speaker_dict.items():

        subfolder_path = os.path.join(folder_path, speaker)
        if os.path.isdir(subfolder_path):
            wav_files = [f for f in os.listdir(subfolder_path) if f.endswith('.wav')]
            if len(wav_files) >= 4:
                selected_files = random.sample(wav_files, 4)
                audios = []
                for file in selected_files:
                    file_path = os.path.join(subfolder_path, file)
                    audio, sr = librosa.load(file_path, mono=True, sr=44100)
                    audio = remove_silence(audio, sr)
                    audios.append(audio)
                    print(f"Loaded {file_path} with sample rate {sr}")
            else:
                print(f"Not enough .wav files in {subfolder_path}")
            
            audios = np.concatenate(audios)
            speaker_audio = torch.tensor(audios[:131072]).unsqueeze(0).unsqueeze(0)
            speaker_audio = pqmf(speaker_audio)
            speaker_emb = speaker_encoder(speaker_audio)[0].detach().cpu().numpy()
        
        print(features["gmm_emb"].sample()[0][0].shape)
        print(features["avg_emb"].shape)
        print(speaker_emb.shape)
        print(features["f0_mean"])
        print(features["f0_std"])

        print(features["f0_mean"])
        
        json_ready_data[speaker] = {
            "gmm_emb": features["gmm_emb"].sample()[0][0].tolist(),
            "avg_emb": features["avg_emb"].tolist(),
            "one_emb": speaker_emb.tolist(),
            "f0_mean": features["f0_mean"],
            "f0_std": features["f0_std"]
        }
    
    # Save to JSON
    with open("speaker_dict.json", "w") as f:
        json.dump(json_ready_data, f, indent=2)
