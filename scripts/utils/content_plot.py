import math
import os

import torch
from typing import Tuple
import argparse
from scipy.io import wavfile 

torch.set_grad_enabled(False)

import cached_conv as cc
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags
import librosa
from rave.rave_model import RAVE

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import rave.blocks
import rave.resampler

import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def extract_content_emb(speakers, encoder, pqmf):
    embeddings = []
    labels = []
    for file in speakers:
        speaker_id = file.split('/')[-2]
        audio, sr = librosa.load(file, sr=44100)
        emb_audio = torch.tensor(audio[:131072]).unsqueeze(0).unsqueeze(0)
        if emb_audio.shape[-1] < 131072:
            padding = torch.zeros(1, 1, 131072 - emb_audio.shape[-1])
            emb_audio = torch.cat([emb_audio, padding], dim=-1)
        audio_multiband = pqmf(emb_audio)
        embedding = encoder(audio_multiband[:, :6, :])
        for i in range(embedding.shape[-1]):
            frame = embedding[:, :, i]
            embeddings.append(frame.detach().cpu().numpy())
            labels.append(speaker_id)
    embeddings = np.array(embeddings).reshape(-1, 64)
    return embeddings, labels


def reduce_and_plot_tsne(embeddings, labels, name):
    print("Reducing dimensions with t-SNE...")
                    
    fig, ax = plt.subplots()
        
    # Create a dictionary to map unique speaker IDs to colors
    unique_speakers = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_speakers))
    color_dict = {speaker: colors(i) for i, speaker in enumerate(unique_speakers)}
    
    tsne = TSNE(n_components=2, random_state=42)
    components_tsne = tsne.fit_transform(embeddings)
    
    for i, speaker in enumerate(unique_speakers):
        indices = [j for j, s in enumerate(labels) if s == speaker]
        ax.scatter(components_tsne[indices, 0], components_tsne[indices, 1], label=speaker, color=color_dict[speaker])
    
    # Display a legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{name}', bbox_inches='tight')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Path to the folder to run", required=True)
    args = parser.parse_args()

    name = args.run.split('/')[1]
    
    cc.use_cached_conv(True)

    generator = RAVE()

    kwargs = {
            "folder": f"{args.run}",
            "map_location": "cpu",
            "package": False,
        }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(torch.device('cpu'))
    generator.eval()

    stereo = False
    sample_rate = generator.sample_rate

    x = torch.zeros(1, 1, 2**17).to(torch.device('cpu'))
    p = torch.zeros(1, 128).to(torch.device('cpu'))
    y = generator.predict_no_pitch(x, x, p)
    print("Shape of test output:", y.shape)

    phrase1 = "../vctk-small/p225/p225_003_mic1.flac"
    phrase2 = "../vctk-small/p226/p226_003_mic1.flac"
    phrase3 = "../vctk-small/p227/p227_003_mic1.flac"
    phrase4 = "../vctk-small/p228/p228_003_mic1.flac"
    phrases = [phrase1, phrase2, phrase3, phrase4]

    encoder = generator.encoder
    speaker_encoder = generator.speaker_encoder
    pqmf = generator.pqmf
    
    print("Processing content embeddings...")
    frames, labels = extract_content_emb(phrases, encoder, pqmf)
    reduce_and_plot_tsne(frames, labels, name="plots/content_frames.png")
    
if __name__ == "__main__":
    main()