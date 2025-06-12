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
    Returns both a dictionary for speaker-specific processing and a flattened list.
    """
    speaker_files = {}
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
            
            speaker_files[speaker_id] = selected_files
            all_files.extend(selected_files)
            print(f"Selected {len(selected_files)} files for speaker {speaker_id}")
    
    return speaker_files, all_files

def extract_speaker_emb(file_paths, speaker_encoder, pqmf):
    """
    Extract speaker embeddings from audio files using 44kHz processing.
    """
    embeddings = []
    labels = []
    
    for speaker_id, files in tqdm(file_paths.items(), desc="Processing Speaker Embeddings"):
        for file in files:
            try:
                audio, sr = librosa.load(file, sr=44100)
                
                # Pad or truncate to 131072 samples
                if len(audio) < 131072:
                    audio = np.pad(audio, (0, 131072 - len(audio)))
                else:
                    audio = audio[:131072]
                
                emb_audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).float()
                audio_multiband = pqmf(emb_audio)
                embedding = speaker_encoder(audio_multiband)
                embeddings.append(embedding.detach().cpu().numpy().flatten())
                labels.append(speaker_id)
                
            except Exception as e:
                print(f"Error processing speaker file {file}: {e}")
    
    return np.array(embeddings), labels

def extract_content_emb_frames(files, encoder):
    """
    Extract frame-level content embeddings from audio files using 16kHz processing.
    This extracts individual frames for detailed content analysis.
    """
    embeddings = []
    labels = []
    
    for file in tqdm(files, desc="Processing Content Embeddings (Frames)"):
        speaker_id = file.split('/')[-2]
        
        try:
            audio, sr = librosa.load(file, sr=16000)
            
            # Pad or truncate to 32768 samples
            if len(audio) < 32768:
                audio = np.pad(audio, (0, 32768 - len(audio)))
            else:
                audio = audio[:32768]
            
            emb_audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).float()
            embedding = encoder(emb_audio)
            
            # Extract each frame
            for i in range(embedding.shape[-1]):
                frame = embedding[:, :, i]
                embeddings.append(frame.detach().cpu().numpy().flatten())
                labels.append(speaker_id)
                
        except Exception as e:
            print(f"Error processing content file {file}: {e}")
    
    return np.array(embeddings), labels

def extract_content_emb_mean(files, encoder):
    """
    Extract mean content embeddings from audio files using 16kHz processing.
    This averages across time for utterance-level representation.
    """
    embeddings = []
    labels = []
    
    for file in tqdm(files, desc="Processing Content Embeddings (Mean)"):
        speaker_id = file.split('/')[-2]
        
        try:
            audio, sr = librosa.load(file, sr=16000)
            
            # Pad or truncate to 32768 samples
            if len(audio) < 32768:
                audio = np.pad(audio, (0, 32768 - len(audio)))
            else:
                audio = audio[:32768]
            
            emb_audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0).float()
            z = encoder(emb_audio)
            
            # Get mean embedding across time dimension
            emb = torch.mean(z, dim=2)
            embeddings.append(emb.detach().cpu().numpy().flatten())
            labels.append(speaker_id)
            
        except Exception as e:
            print(f"Error processing content file {file}: {e}")
    
    return np.array(embeddings), labels

def plot_tsne(embeddings, labels, ax, title):
    """
    Reduce dimensions with t-SNE and plot on given axis.
    """
    print(f"Processing t-SNE for {title}...")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Create color mapping
    unique_speakers = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_speakers))
    color_dict = {speaker: colors(i) for i, speaker in enumerate(unique_speakers)}
    
    # Adjust perplexity based on number of samples
    perplexity = min(30, len(embeddings) // 3, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    components_tsne = tsne.fit_transform(scaled_embeddings)
    
    # Plot each speaker
    for speaker in unique_speakers:
        indices = [j for j, s in enumerate(labels) if s == speaker]
        ax.scatter(components_tsne[indices, 0], components_tsne[indices, 1], 
                  label=speaker, color=color_dict[speaker], alpha=0.7, s=30)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Path to the folder to run", required=True)
    parser.add_argument("--speakers_dir", type=str, help="Path to the speakers directory", required=True)
    parser.add_argument("--num_files", type=int, default=50, help="Number of files per speaker")
    args = parser.parse_args()

    model_name = args.run.split('/')[-3]
    print(model_name)
    
    print(f"Loading VoiceModel from {args.run}...")
    cc.use_cached_conv(True)

    generator = VoiceModel()
    kwargs = {
        "folder": args.run,
        "map_location": 'cpu',
        "package": False
    }

    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(torch.device('cpu'))
    generator.eval()

    sample_rate = generator.sample_rate
    print(f"Model sample rate: {sample_rate}")

    # Test the model
    x = torch.zeros(1, 1, 2**15).to(torch.device('cpu'))
    try:
        y = generator(x)
        print("Model test successful. Output shape:", y['audio'].shape)
    except Exception as e:
        print(f"Model test failed: {e}")
        return

    # Extract model components
    encoder = generator.encoder
    speaker_encoder = generator.speaker_encoder
    pqmf = generator.pqmf
    
    # Get files
    print(f"Getting files from {args.speakers_dir}...")
    speaker_files, all_files = get_random_files(args.speakers_dir, x=args.num_files)
    print(f"Total files selected: {len(all_files)}")
    
    if not all_files:
        print("No files found. Exiting.")
        return
    
    # Process specific test phrases (if they exist)
    test_phrases = [
        "../vctk-small/p225/p225_003_mic1.flac",
        "../vctk-small/p226/p226_003_mic1.flac",
        "../vctk-small/p227/p227_003_mic1.flac",
        "../vctk-small/p228/p228_003_mic1.flac"
    ]
    
    # Filter existing test phrases
    existing_phrases = [p for p in test_phrases if os.path.exists(p)]
    if not existing_phrases:
        existing_phrases = all_files[:4]  # Use first 4 files as fallback
    
    print("Extracting embeddings...")
    
    # Extract all embeddings
    speaker_embeddings, speaker_labels = extract_speaker_emb(speaker_files, speaker_encoder, pqmf)
    content_frames, content_frame_labels = extract_content_emb_frames(existing_phrases, encoder)
    content_mean, content_mean_labels = extract_content_emb_mean(all_files, encoder)
    
    print(f"Speaker embeddings: {speaker_embeddings.shape}")
    print(f"Content frames: {content_frames.shape}")
    print(f"Content mean: {content_mean.shape}")
    
    # Create subplot visualization - 3 plots side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Voice Model Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot three analyses side by side
    plot_tsne(speaker_embeddings, speaker_labels, axes[0], 
              'Speaker Embeddings (Identity Clustering)')
    
    plot_tsne(content_frames, content_frame_labels, axes[1], 
              'Content Embeddings - Frame Level (Test Phrases)')
    
    plot_tsne(content_mean, content_mean_labels, axes[2], 
              'Content Embeddings - Mean Level (All Files)')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plot_filename = f'plots/voice_analysis_{model_name}.png'
    plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
    print(f"Analysis plot saved as {plot_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()