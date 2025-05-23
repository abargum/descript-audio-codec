"""
DNSMOS quality evaluation for processed audio.
"""

import os
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

def calculate_dnsmos_scores(processed_folder_44, targets):
    """
    Calculate DNSMOS scores for all processed audio files in the 44kHz folder.
    
    Args:
        processed_folder_44 (str): Path to the folder containing processed audio files at 44.1kHz
        targets (list): List of target speaker IDs
    
    Returns:
        dict: Dictionary with DNSMOS scores for each speaker
    """
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize DNSMOS model
        dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=44100, personalized=False, device=device)
    except Exception as e:
        print(f"Error initializing DNSMOS model: {e}")
        return {}
    
    for speaker in tqdm(targets, desc="Calculating DNSMOS scores"):
        speaker_scores = {
            'bak': [],
            'sig': [],
            'ovrl': []
        }
        
        processed_speaker_path = os.path.join(processed_folder_44, speaker)
        
        if not os.path.exists(processed_speaker_path):
            print(f"Warning: Processed folder for {speaker} not found at {processed_speaker_path}")
            continue
            
        # Get all processed files for this speaker
        processed_files = list(Path(processed_speaker_path).glob("**/*.wav"))
        
        if not processed_files:
            print(f"Warning: No processed files found for {speaker}")
            continue
            
        # Calculate DNSMOS for each processed file
        for processed_file in tqdm(processed_files, desc=f"Processing {speaker} files", leave=False):
            try:
                # Load audio file
                y, sr = librosa.load(str(processed_file), sr=44100)
                
                # Check if audio is empty
                if len(y) == 0:
                    print(f"Warning: Empty audio file {processed_file}, skipping...")
                    continue
                
                # Convert to tensor
                audio_tensor = torch.tensor(y, device=device).float()
                
                # Calculate DNSMOS
                scores = dnsmos(audio_tensor)
                
                # Extract scores
                speaker_scores['sig'].append(scores[0].item())
                speaker_scores['bak'].append(scores[1].item())
                speaker_scores['ovrl'].append(scores[2].item())
                
            except Exception as e:
                print(f"Error processing {processed_file}: {e}")
        
        # Calculate mean scores
        results[speaker] = {
            'mean_bak': np.mean(speaker_scores['bak']) if speaker_scores['bak'] else 0,
            'mean_sig': np.mean(speaker_scores['sig']) if speaker_scores['sig'] else 0,
            'mean_ovrl': np.mean(speaker_scores['ovrl']) if speaker_scores['ovrl'] else 0,
            'individual_scores': speaker_scores
        }
    
    return results

def print_dnsmos_report(dnsmos_results):    
    """Print a formatted DNSMOS report."""
    print("\n===== DNSMOS EVALUATION REPORT =====\n")
    
    # Calculate overall averages
    all_bak = []
    all_sig = []
    all_ovrl = []
    
    # Print individual speaker results
    for speaker, data in dnsmos_results.items():
        print(f"\n== Speaker: {speaker} ==")
        print(f"BAK (background quality): {data['mean_bak']:.4f}")
        print(f"SIG (signal quality): {data['mean_sig']:.4f}")
        print(f"OVRL (overall quality): {data['mean_ovrl']:.4f}")
        
        all_bak.append(data['mean_bak'])
        all_sig.append(data['mean_sig'])
        all_ovrl.append(data['mean_ovrl'])
    
    # Print overall averages
    print("\n== Overall Average Scores ==")
    print(f"Average BAK: {np.mean(all_bak):.4f}")
    print(f"Average SIG: {np.mean(all_sig):.4f}")
    print(f"Average OVRL: {np.mean(all_ovrl):.4f}")
    
    print("\n===========================================")