import sys
import os
import time
import json
import shutil
import argparse
import torch
import random
from sklearn import mixture
import pickle

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from rave.blocks3 import SpeakerRAVE
from rave.pqmf import CachedPQMF as PQMF
from utils import load_dict_from_txt, load_speaker_statedict

import librosa
import numpy as np
from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda:0")

def get_f0(audio):
    f0 = pitch_model.infer(audio, sr=44100, decoder_mode='local_argmax', threshold=0.006, f0_min=50, f0_max=750)
    f0_nonzero = f0[f0 > 0]
    return f0_nonzero

def get_emb_and_f0(path, model, pqmf):
    x, _ = librosa.load(path, sr=44100, mono=True)
    x = torch.tensor(x[:131072]).unsqueeze(0).float()
    if x.shape[-1] < 131072:
        pad = torch.zeros(1, 131072 - x.shape[-1])
        x = torch.cat((x, pad), dim=-1)

    f0 = get_f0(x)
    
    x_pqmf = pqmf(x.unsqueeze(1).to('cuda'))
    embed = model(x_pqmf)

    embed = embed.detach().cpu().squeeze().numpy()
    return embed, f0
    

def create_embeddings(folder_path, model, pqmf):

    speaker_emb_dict = {}
    
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for i, subfolder in enumerate(subfolders):
        speaker_id = os.path.basename(subfolder)
        print(f"Calculating embeddings for speaker {speaker_id} - {i+1} of {len(subfolders)}")
        audio_files = [
            os.path.join(subfolder, file)
            for file in os.listdir(subfolder)
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'))
        ]
        # Select up to num_files audio files from the current subfolder
        if len(audio_files) > 0:
            utt_embeddings = []
            utt_f0s = []
            
            for file in audio_files:
                utt_emb, f0 = get_emb_and_f0(file, model, pqmf)
                utt_embeddings.append(utt_emb)
                utt_f0s.append(f0)

            utt_f0s = torch.cat(utt_f0s)                
            utt_embeddings = np.stack(utt_embeddings)
            gmm_dvector = mixture.GaussianMixture(n_components=1, covariance_type="diag")
            gmm_dvector.fit(utt_embeddings)

        speaker_emb_dict[speaker_id] = {}
        speaker_emb_dict[speaker_id]['gmm_emb'] = gmm_dvector
        speaker_emb_dict[speaker_id]['avg_emb'] = np.mean(utt_embeddings, axis=0)
        speaker_emb_dict[speaker_id]['f0_mean'] = torch.mean(utt_f0s).item()
        speaker_emb_dict[speaker_id]['f0_std']  = torch.std(utt_f0s).item()
        
    return speaker_emb_dict


def save_embeddings(embs, path):
    with open(path, 'wb') as f:
        pickle.dump(embs, f)
        print(f"Speaker embedding saved to {path}.")


def main():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--dataset_path', type=str, default='VCTK-Corpus/wav48')

    args = parser.parse_args()

    file = 'VCTK-Corpus/speaker-info.txt'
    info_dict = load_dict_from_txt(file)

    pqmf = PQMF(attenuation = 100, n_band = 16)

    speaker_encoder = SpeakerRAVE()
    spk_state, pqmf_state = load_speaker_statedict("scripts/rave/model000000075.model")
    speaker_encoder.load_state_dict(spk_state)
    speaker_encoder.eval()
    
    pqmf.to('cuda')
    speaker_encoder.to('cuda')
    
    speaker_emb_dict = create_embeddings(args.dataset_path, speaker_encoder, pqmf)

    save_embeddings(speaker_emb_dict, 'scripts/utils/speaker_emb_dict.pkl')

if __name__ == "__main__":
    main()