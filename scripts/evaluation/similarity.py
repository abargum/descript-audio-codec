"""
Voice similarity evaluation using Resemblyzer.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from resemblyzer import preprocess_wav, VoiceEncoder

def calculate_similarity_scores(processed_folder, target_speaker_folder, targets, encoder=None):
    """
    Calculate resemblyzer similarity scores between processed audio and target speakers.
    
    Args:
        processed_folder (str): Path to the folder with processed audio files
        target_speaker_folder (str): Path to the folder with target speaker audio files
        targets (list): List of target speaker IDs
        encoder (VoiceEncoder, optional): Pre-initialized voice encoder
    
    Returns:
        dict: Dictionary with similarity scores for each speaker pair
    """
    
    # Initialize encoder if not provided
    if encoder is None:
        encoder = VoiceEncoder()
    
    results = {}
    
    # Check if target speaker folder exists
    if not os.path.exists(target_speaker_folder):
        raise FileNotFoundError(f"Target speaker folder not found: {target_speaker_folder}")
    
    # Process target speakers to get their embeddings
    target_embeddings = {}
    for speaker in tqdm(targets, desc="Processing target speakers"):
        speaker_wavs = []
        speaker_path = os.path.join(target_speaker_folder, speaker)
        
        if not os.path.exists(speaker_path):
            print(f"Warning: Target speaker folder not found: {speaker_path}")
            continue
        
        # Get all WAV files for this target speaker
        wav_files = list(Path(speaker_path).glob("**/*.wav"))
        if not wav_files:
            wav_files = list(Path(speaker_path).glob("**/*.flac"))
        
        if not wav_files:
            print(f"Warning: No audio files found for target speaker {speaker}")
            continue
            
        # Preprocess all wavs for this speaker
        for wav_path in wav_files:
            try:
                preprocessed_wav = preprocess_wav(str(wav_path))
                speaker_wavs.append(preprocessed_wav)
            except Exception as e:
                print(f"Error preprocessing {wav_path}: {e}")
        
        if speaker_wavs:
            # Create an embedding for the target speaker using all available samples
            target_embeddings[speaker] = encoder.embed_speaker(speaker_wavs)
    
    # Calculate similarity for each processed audio against its target
    for speaker in tqdm(targets, desc="Calculating similarity scores"):
        speaker_scores = []
        processed_speaker_path = os.path.join(processed_folder, speaker)
        
        if not os.path.exists(processed_speaker_path):
            print(f"Warning: Processed folder for {speaker} not found at {processed_speaker_path}")
            continue
            
        # Get all processed files for this speaker
        processed_files = list(Path(processed_speaker_path).glob("**/*.wav"))
        
        if not processed_files:
            print(f"Warning: No processed files found for {speaker}")
            continue
            
        # Calculate similarity for each processed file
        for processed_file in tqdm(processed_files, desc=f"Processing {speaker} files", leave=False):
            try:
                preprocessed_wav = preprocess_wav(str(processed_file))
                processed_embedding = encoder.embed_utterance(preprocessed_wav)
                
                # Compare to target embedding
                if speaker in target_embeddings:
                    similarity = np.inner(processed_embedding, target_embeddings[speaker])
                    rel_path = os.path.relpath(processed_file, processed_speaker_path)
                    speaker_scores.append((rel_path, float(similarity)))
            except Exception as e:
                print(f"Error processing {processed_file}: {e}")
        
        # Store results for this speaker
        results[speaker] = {
            'average_similarity': np.mean([score for _, score in speaker_scores]) if speaker_scores else 0,
            'individual_scores': sorted(speaker_scores, key=lambda x: x[1], reverse=True)
        }
    
    return results

def print_similarity_report(similarity_results):    
    """Print a formatted similarity report."""
    print("\n===== VOICE CONVERSION SIMILARITY REPORT =====\n")
    
    # Print individual speaker results
    for speaker, data in similarity_results.items():
        if speaker == 'cross_speaker_similarity':
            continue
            
        print(f"\n== Speaker: {speaker} ==")
        print(f"Average similarity score: {data['average_similarity']:.4f}")
    
    print("\n===========================================")