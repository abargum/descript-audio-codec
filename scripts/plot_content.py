import math
import os
import torch
from typing import Tuple
import argparse
from scipy.io import wavfile 
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import librosa
import numpy as np

torch.set_grad_enabled(False)

import cached_conv as cc
from modules.model import VoiceModel

def get_random_files(folder_path, x=10):
    """
    Select x random audio files from each speaker's folder.
    Returns a flattened list of files with their speaker IDs.
    """
    all_files = []
    for speaker_id in os.listdir(folder_path):
        speaker_path = os.path.join(folder_path, speaker_id)
        if os.path.isdir(speaker_path):
            files = [os.path.join(speaker_path, f) for f in os.listdir(speaker_path) 
                    if f.endswith(('.flac', '.wav'))]
            if len(files) < x:
                print(f"Speaker {speaker_id} has less than {x} files. Using all {len(files)} files.")
                selected_files = files
            else:
                selected_files = random.sample(files, x)
            
            all_files.extend(selected_files)
            print(f"Selected {len(selected_files)} files for speaker {speaker_id}")
    
    return all_files

def extract_content_emb(files, encoder, pqmf, rvq):
    """
    Extract content embeddings from audio files.
    """
    embeddings = []
    labels = []
    
    for file in tqdm(files, desc="Processing audio files"):
        speaker_id = file.split('/')[-2]  # Extract speaker ID from path
        
        try:
            audio, sr = librosa.load(file, sr=44100)
            
            # Ensure we have enough audio data
            if len(audio) < 131072:
                print(f"File {file} is too short ({len(audio)} samples). Padding...")
                audio = np.pad(audio, (0, 131072 - len(audio)))
            else:
                audio = audio[:131072]  # Truncate if needed
            
            # Process through model
            emb_audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).float()
            audio_multiband = pqmf(emb_audio)
            
            # Extract embeddings
            z = encoder(audio_multiband[:, :6, :])
            z, vq_out, rvq_out, _ = rvq(z)
            
            # Get the mean embedding across time dimension
            emb = torch.mean(z, dim=2)
            embeddings.append(emb.detach().cpu().numpy().flatten())
            labels.append(speaker_id)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    return np.array(embeddings), labels

def reduce_and_plot_tsne(embeddings, labels, name):
    """
    Reduce dimensions with t-SNE and plot.
    """
    print("Reducing dimensions with t-SNE...")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    fig, ax = plt.subplots()
    
    # Create a dictionary to map unique speaker IDs to colors
    unique_speakers = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_speakers))
    color_dict = {speaker: colors(i) for i, speaker in enumerate(unique_speakers)}
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    components_tsne = tsne.fit_transform(scaled_embeddings)
    
    for i, speaker in enumerate(unique_speakers):
        indices = [j for j, s in enumerate(labels) if s == speaker]
        ax.scatter(components_tsne[indices, 0], components_tsne[indices, 1], 
                  label=speaker, color=color_dict[speaker], alpha=0.7)
    
    # Display a legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE Visualization of Speaker Content Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{name}', bbox_inches='tight')
    print(f"Plot saved as plots/{name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Path to the folder to run", required=True)
    parser.add_argument("--speakers_dir", type=str, help="Path to the speakers directory", required=True)
    args = parser.parse_args()

    model_name = args.run.split('/')[1]
    
    print(f"Loading model from {args.run}...")
    cc.use_cached_conv(True)

    generator = VoiceModel()

    kwargs = {
        "folder": f"{args.run}",
        "map_location": "cpu",
        "package": False,
    }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(torch.device('cpu'))
    generator.eval()

    sample_rate = generator.sample_rate
    print(f"Model sample rate: {sample_rate}")

    # Test the model with dummy input
    x = torch.zeros(1, 1, 2**17).to(torch.device('cpu'))
    try:
        y = generator.get_val_audio(x)['audio']
        print("Model test successful. Output shape:", y.shape)
    except Exception as e:
        print(f"Model test failed: {e}")

    # Extract relevant components from the model
    encoder = generator.encoder
    rvq = generator.split_rvq
    pqmf = generator.pqmf
    
    # Get random files from speaker directories
    print(f"Getting random files from {args.speakers_dir}...")
    all_files = get_random_files(args.speakers_dir, x=10)
    print(f"Total files selected: {len(all_files)}")
    
    if not all_files:
        print("No files found. Exiting.")
        return
    
    print("Processing content embeddings...")
    embeddings, labels = extract_content_emb(all_files, encoder, pqmf, rvq)
    
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Create t-SNE plot
    plot_name = f"tsne_content_embeddings_{model_name}_rvq.png"
    reduce_and_plot_tsne(embeddings, labels, plot_name)

if __name__ == "__main__":
    main()