import sys
import os
import time
import json
import shutil
import argparse
import torch
import random

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from rave.blocks3 import SpeakerRAVE
from rave.pqmf import CachedPQMF as PQMF
from utils import load_dict_from_txt, load_speaker_statedict

import librosa
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # For color conversion
import pandas as pd
import plotly.express as px

import pandas as pd
import yaml


def plot_audio_embeddings(audio_files, instrument_classes, model, pqmf, info_dict, tsne=False, save_path="plots/embedding_plot.png"):
    speaker_E = []
    speaker_ID = []
    speaker_AGE = []
    speaker_GENDER = []
    
    seconds = 0
    start = seconds * 44100

    model.to('cuda')
    pqmf.to('cuda')
    model.eval()
    
    for file in audio_files:
        for audio in file:
            inst_class = audio.split('/')[2]
            if inst_class != 'p280':

                age = info_dict[str(inst_class)]['AGE']
                gender = info_dict[str(inst_class)]['GENDER']
                
                x, _ = librosa.load(audio, sr=44100, mono=True)
                x = torch.tensor(x[start:start+131072]).unsqueeze(0).float()
                if x.shape[-1] < 131072:
                    pad = torch.zeros(1, 131072 - x.shape[-1])
                    x = torch.cat((x, pad), dim=-1)
    
                x_pqmf = pqmf(x.unsqueeze(1).to('cuda'))
                embed = model(x_pqmf)
                
                speaker_E.append(embed.detach().cpu().numpy())
                speaker_ID.append(inst_class)
                speaker_AGE.append(age)
                speaker_GENDER.append(gender)
    
    speaker_E = np.array(speaker_E).reshape(-1, 256)

    if not tsne:
        pca = PCA(n_components=2)
        components_pca = pca.fit_transform(speaker_E)
    
        # Calculate IQR for each PCA dimension
        q1 = np.percentile(components_pca, 25, axis=0)
        q3 = np.percentile(components_pca, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Determine indices within the bounds for all dimensions
        valid_indices = np.all((components_pca >= lower_bound) & (components_pca <= upper_bound), axis=1)
        
        filtered_components_pca = components_pca[valid_indices]
        filtered_speaker_ID = [speaker_ID[i] for i in range(len(speaker_ID)) if valid_indices[i]]
        filtered_speaker_AGE = [speaker_AGE[i] for i in range(len(speaker_AGE)) if valid_indices[i]]
        filtered_speaker_GENDER = [speaker_GENDER[i] for i in range(len(speaker_GENDER)) if valid_indices[i]]
    else:
        tsne = manifold.TSNE(n_components=2, random_state=42)
        components_tsne = tsne.fit_transform(speaker_E)
    
        filtered_components_pca = components_tsne
        filtered_speaker_ID = speaker_ID
        filtered_speaker_AGE = speaker_AGE
        filtered_speaker_GENDER = speaker_GENDER

    # Set up the plot
    fig, ax = plt.subplots()
    unique_speakers = list(set(filtered_speaker_ID))
    colors = plt.cm.get_cmap('tab20', len(unique_speakers))
    color_dict = {speaker: colors(i) for i, speaker in enumerate(unique_speakers)}
    
    for i, speaker in enumerate(unique_speakers):
        indices = [j for j, s in enumerate(filtered_speaker_ID) if s == speaker]
        ax.scatter(filtered_components_pca[indices, 0], filtered_components_pca[indices, 1], label=speaker, color=color_dict[speaker])
    
    # Set labels and legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return filtered_components_pca, filtered_speaker_ID, filtered_speaker_AGE, filtered_speaker_GENDER


def plot_interactive_pca(components_pca, speaker_ID, speaker_AGE, speaker_GENDER, save_path="plots/interactive_embedding_plot.html"):
    # Create a DataFrame for the PCA components and speaker IDs
    df = pd.DataFrame({
        'Principal Component 1': components_pca[:, 0],
        'Principal Component 2': components_pca[:, 1],
        'Speaker': speaker_ID
    })

    # Add Age and Gender information to the DataFrame
    df['Age'] = [speaker_AGE[i] for i in range(len(speaker_ID))]
    df['Gender'] = [speaker_GENDER[i] for i in range(len(speaker_ID))]

    # Generate unique colors for each speaker
    unique_speakers = sorted(set(speaker_ID))
    num_speakers = len(unique_speakers)
    cmap = plt.cm.get_cmap("tab20", num_speakers)  # "tab20" colormap
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(num_speakers)]  # Convert to hex colors
    color_map = {speaker: colors[i] for i, speaker in enumerate(unique_speakers)}  # Map speakers to colors

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        df,
        x='Principal Component 1',
        y='Principal Component 2',
        color='Speaker',
        hover_data=['Speaker', 'Age', 'Gender'],
        title='Enhanced Interactive PCA Embedding',
        color_discrete_map=color_map  # Use the unique color mapping
    )

    fig.update_traces(marker=dict(size=9))  # Double the size (default is 6)

    # Save the plot to an HTML file
    fig.write_html(save_path)
    print(f"Interactive plot saved to {save_path}")

    
def get_random_audio_files_from_subfolders(folder_path, num_files=15, max_speakers=10):
    selected_files = []
    instrument_classes = []
    
    # Get a list of subfolders in the root folder
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # Randomly choose up to max_speakers subfolders
    chosen_subfolders = random.sample(subfolders, min(max_speakers, len(subfolders)))
    
    for subfolder in chosen_subfolders:
        inst_class = os.path.basename(subfolder)
        instrument_classes.append(inst_class)
        audio_files = [
            os.path.join(subfolder, file)
            for file in os.listdir(subfolder)
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'))
        ]
        # Select up to num_files audio files from the current subfolder
        if len(audio_files) > 0:
            selected_files.append(random.sample(audio_files, min(num_files, len(audio_files))))

    return selected_files, instrument_classes

def main():
    random.seed(18)
    
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
    
    audio_files, instrument_classes = get_random_audio_files_from_subfolders(args.dataset_path, num_files=30, max_speakers=20)
    filtered_components_pca, filtered_speaker_ID, filtered_speaker_AGE, filtered_speaker_GENDER = plot_audio_embeddings(audio_files, instrument_classes, speaker_encoder, pqmf, info_dict)
    plot_interactive_pca(filtered_components_pca, filtered_speaker_ID, filtered_speaker_AGE, filtered_speaker_GENDER)


if __name__ == "__main__":
    main()