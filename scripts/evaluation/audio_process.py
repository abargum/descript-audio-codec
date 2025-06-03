"""
Audio processing utilities for voice conversion evaluation.
"""

import os
import numpy as np
import librosa
import torch
from scipy.io import wavfile
from pathlib import Path
from tqdm import tqdm

def adjust_audio_length(y, sr, min_power=14, mode="truncate"):
    """
    Adjust audio length to a power of 2.
    
    Args:
        y (np.array): Audio signal
        sr (int): Sample rate
        min_power (int): Minimum power of 2
        mode (str): 'truncate' or 'pad'
    
    Returns:
        np.array: Adjusted audio signal
    """
    current_length = len(y)
    min_length = 2 ** min_power
    target_length = 2 ** int(np.ceil(np.log2(current_length))) 
    lower_power = 2 ** int(np.floor(np.log2(current_length)))

    if mode == "truncate":
        target_length = max(lower_power, min_length)
        y_adj = y[:target_length]
    elif mode == "pad":
        target_length = max(target_length, min_length)
        y_adj = np.pad(y, (0, target_length - current_length), mode='constant')
    else:
        raise ValueError("Invalid mode. Choose 'truncate' or 'pad'.")
    
    return y_adj

def find_audio_files(input_folder):
    """
    Find all audio files in the input folder.
    
    Args:
        input_folder (str): Path to input folder
    
    Returns:
        list: List of audio file paths
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input audio folder not found: {input_folder}")
    
    audio_files = []
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {input_folder}")
    
    return audio_files

def process_single_audio_file(audio_path, target, embeddings, means, stds, 
                            input_folder, output_folder, processed_folder,
                            generator, device, min_power=14, mode="pad"):
    """
    Process a single audio file for a specific target speaker.
    
    Args:
        audio_path (str): Path to the audio file
        target (str): Target speaker ID
        embeddings, means, stds: Speaker-specific parameters
        input_folder, output_folder, processed_folder: Folder paths
        generator: Voice conversion model
        device (str): Device to run inference on
        min_power (int): Minimum power of 2 for audio length
        mode (str): Audio adjustment mode
    """
    try:
        processed_folder_44 = processed_folder + "_44"
        
        rel_path = os.path.relpath(audio_path, input_folder)                
        output_audio_path = os.path.join(output_folder, target, rel_path)
        processed_audio_path = os.path.join(processed_folder, target, rel_path)
        processed_audio_path_44 = os.path.join(processed_folder_44, target, rel_path)
        
        # Create output directories
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_audio_path_44), exist_ok=True)
        
        # Load and process audio
        y, sr = librosa.load(audio_path, sr=44100)
        y = librosa.util.normalize(y, axis=-1)
        
        # Check if audio is too short or empty
        if len(y) == 0:
            print(f"Warning: Empty audio file {audio_path}, skipping...")
            return
        
        y_adj = adjust_audio_length(y, sr, min_power=min_power, mode=mode)
        
        # Resample to 16kHz for processing
        y_16k = librosa.resample(y_adj.astype(np.float32), orig_sr=sr, target_sr=16000)

        # Save resampled input
        output_audio_path = output_audio_path.replace(".flac", ".wav")
        wavfile.write(output_audio_path, 16000, y_16k)
        
        # Process with model
        audio_tensor = torch.tensor(y_adj).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            processed = generator.evaluate(audio_tensor, embeddings, means, stds)

        # Save processed 44.1 kHz version
        processed_np = processed.squeeze().cpu().numpy()
        processed_audio_path_44 = processed_audio_path_44.replace(".flac", ".wav")
        wavfile.write(processed_audio_path_44, 44100, processed_np.astype(np.float32))

        # Save processed 16kHz version
        processed_np_16k = librosa.resample(processed_np.astype(np.float32), 
                                          orig_sr=44100, target_sr=16000)
        processed_audio_path = processed_audio_path.replace(".flac", ".wav")
        wavfile.write(processed_audio_path, 16000, processed_np_16k.astype(np.float32))
        
        return rel_path
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_audio_files(generator, targets, embeddings, means, stds, 
                       input_folder, output_folder, processed_folder, device,
                       min_power=14, mode="pad"):
    """
    Process all audio files for all target speakers.
    
    Args:
        generator: Voice conversion model
        targets (list): List of target speaker IDs
        embeddings, means, stds: Speaker-specific parameters
        input_folder (str): Input audio folder
        output_folder (str): Output folder for resampled audio
        processed_folder (str): Output folder for processed audio
        device (str): Device for inference
        min_power (int): Minimum power of 2 for audio length
        mode (str): Audio adjustment mode
    """
    processed_folder_44 = processed_folder + "_44"
    
    # Create main output directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(processed_folder_44, exist_ok=True)
    
    # Find all audio files
    audio_files = find_audio_files(input_folder)
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file for each target speaker
    total_processed = 0
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        for i, target in enumerate(targets):
            result = process_single_audio_file(
                audio_path=audio_path,
                target=target,
                embeddings=embeddings[i],
                means=means[i],
                stds=stds[i],
                input_folder=input_folder,
                output_folder=output_folder,
                processed_folder=processed_folder,
                generator=generator,
                device=device,
                min_power=min_power,
                mode=mode
            )
            
            if result:
                total_processed += 1
                if total_processed % 10 == 0:  # Print progress every 10 files
                    print(f"Processed: {result}")
    
    print(f"Processing complete. Successfully processed {total_processed} file-speaker pairs.")