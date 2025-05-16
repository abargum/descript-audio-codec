import cached_conv as cc
import librosa
import torch
import IPython.display as ipd
from scipy.io import wavfile 
import sys
import os
from scipy.io import wavfile
import argparse
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
import jiwer
import shutil
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from modules.model import VoiceModel
from modules.utils import get_f0_fcpe, extract_f0_mean_std
from export_utils.pitchTracker import PitchRegisterTracker
from utils.utils import load_dict_from_txt

parser = argparse.ArgumentParser(description='Voice model processing script')
parser.add_argument('--model', type=str, help='Path to the model folder')
parser.add_argument('--input_audio_folder', type=str, default="audio", help='Path to the input audio folder')
parser.add_argument('--resampled_audio_folder', type=str, default="scripts/evaluation/resampled", help='Path to the resampled input audio folder')
parser.add_argument('--processed_audio_folder', type=str, default="scripts/evaluation/processed",  help='Path to the processed audio folder')
parser.add_argument('--target_speaker_folder', type=str, default="vctk-small",  help='Path to the target audio folder')
parser.add_argument('--target_sr', type=int, default=44100, help='Saving sample rate for any metrics')
parser.add_argument('--device', type=str, default="cuda", help='Device to run the model on (cuda or cpu)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

def set_seed(seed):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for CUDA
    torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior on GPU
    torch.backends.cudnn.benchmark = False  # Disable optimizations for non-deterministic algorithms
    print(f"Random seed {seed} has been set for reproducibility.")

def adjust_audio_length(y, sr, min_power=14, mode="truncate"):
    """
    Adjust audio length to a power of 2.
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

def get_speaker_embeddings(targets):
    file_path = 'scripts/utils/speaker_emb_dict.pkl'
    with open(file_path, 'rb') as file:
        speaker_dict = pickle.load(file)
    
    emb_list = nn.ParameterList()
    f0_mean_list = []
    f0_std_list = []
    
    for speaker in targets:
        target_stats = speaker_dict[speaker]
        target_emb = torch.tensor(target_stats['avg_emb']).unsqueeze(0)
        target_f0_mean = target_stats['f0_mean']
        target_f0_std = target_stats['f0_std']
    
        emb_list.append(nn.Parameter(target_emb))
        f0_mean_list.append(target_f0_mean)
        f0_std_list.append(target_f0_std)
    
    return emb_list, f0_mean_list, f0_std_list

def process_audio_files(generator, targets, embeddings, means, stds, input_folder, output_folder, processed_folder, out_sr=44100, min_power=14, mode="truncate"):
    """
    Process all audio files in input_folder (including subfolders) and save both original and processed 
    versions to their respective output folders while maintaining the same folder structure.
    
    Args:
        input_folder (str): Path to the folder containing audio files to process
        output_folder (str): Path to save the resampled input files
        processed_folder (str): Path to save the processed output files
        sample_rate (int): Target sample rate for audio files
        min_power (int): Minimum power of 2 for audio length adjustment
        mode (str): Mode for audio length adjustment ('truncate' or 'pad')
    """
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    # Get all audio files with .wav, .mp3, etc. extensions
    audio_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
                audio_files.append(os.path.join(root, file))
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        for i, target in enumerate(targets):
            try:
                rel_path = os.path.relpath(audio_path, input_folder)                
                output_audio_path = os.path.join(output_folder, target, rel_path)
                processed_audio_path = os.path.join(processed_folder, target, rel_path)
                
                os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
                os.makedirs(os.path.dirname(processed_audio_path), exist_ok=True)
                
                y, sr = librosa.load(audio_path, sr=44100)
                y_adj = adjust_audio_length(y, sr, min_power=min_power, mode=mode)
                
                if sr != out_sr:
                    y = librosa.resample(y_adj.astype(np.float32), orig_sr=sr, target_sr=out_sr)

                output_audio_path = output_audio_path.replace(".flac", ".wav")
                wavfile.write(output_audio_path, out_sr, y)
                
                audio_tensor = torch.tensor(y_adj).unsqueeze(0).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    processed = generator.evaluate(audio_tensor, embeddings[i], means[i], stds[i])
                    #processed = generator.get_val_audio(audio_tensor)["audio"]
                
                processed_np = processed.squeeze().cpu().numpy()
                if sr != out_sr:
                    processed_np = librosa.resample(processed_np.astype(np.float32), orig_sr=sr, target_sr=out_sr)

                processed_audio_path = processed_audio_path.replace(".flac", ".wav")
                wavfile.write(processed_audio_path, out_sr, processed_np.astype(np.float32))
                
                print(f"Processed: {rel_path}")
            
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
    
    print(f"Processing complete. Processed {len(audio_files)} files.")


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
    
    # Process target speakers to get their embeddings
    target_embeddings = {}
    for speaker in tqdm(targets, desc="Processing target speakers"):
        speaker_wavs = []
        speaker_path = os.path.join(target_speaker_folder, speaker)
        
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
                processed_wav = preprocess_wav(str(processed_file))
                processed_embedding = encoder.embed_utterance(processed_wav)
                
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
    print("\n===== VOICE CONVERSION SIMILARITY REPORT =====\n")
    
    # Print individual speaker results
    for speaker, data in similarity_results.items():
        if speaker == 'cross_speaker_similarity':
            continue
            
        print(f"\n== Speaker: {speaker} ==")
        print(f"Average similarity score: {data['average_similarity']:.4f}")
    
    print("\n===========================================")


def calculate_wer(targets, resampled_audio_folder, processed_audio_folder):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device)
    
    # Define jiwer transforms for text normalization
    transforms = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    
    # Store all speaker results for final summary
    all_speaker_results = {}
    
    for speaker in tqdm(targets, desc="Processing speakers"):
        print(f"\n===== Processing speaker: {speaker} =====")
        
        # Get speaker paths
        resampled_speaker_path = os.path.join(resampled_audio_folder, speaker)
        processed_speaker_path = os.path.join(processed_audio_folder, speaker)
        
        # Check if both paths exist
        if not os.path.exists(resampled_speaker_path):
            print(f"Resampled folder for {speaker} not found at {resampled_speaker_path}")
            continue
        if not os.path.exists(processed_speaker_path):
            print(f"Processed folder for {speaker} not found at {processed_speaker_path}")
            continue
        
        # Create dictionaries mapping filenames to full paths
        resampled_files = {f.name: f for f in Path(resampled_speaker_path).glob("**/*.wav")}
        processed_files = {f.name: f for f in Path(processed_speaker_path).glob("**/*.wav")}
        
        # Find files that exist in both folders (same filename)
        common_filenames = set(resampled_files.keys()) & set(processed_files.keys())
        
        if not common_filenames:
            print(f"No matching filenames found for speaker {speaker}")
            continue
        
        print(f"Found {len(common_filenames)} matching files for speaker {speaker}")
        
        # Calculate WER for each matching file
        wer_scores = []
        file_results = []
        
        for filename in tqdm(sorted(common_filenames), desc=f"Calculating WER for {speaker}", leave=False):
            resampled_file = resampled_files[filename]
            processed_file = processed_files[filename]
            
            try:
                # Get ASR results for both versions
                resampled_result = pipe(str(resampled_file))
                processed_result = pipe(str(processed_file))
                
                # Get transcriptions
                resampled_text = resampled_result['text']
                processed_text = processed_result['text']
                
                # Print both transcriptions
                print(f"\nFile: {filename}")
                print(f"Resampled: {resampled_text}")
                print(f"Processed: {processed_text}")
                
                # Calculate WER
                wer = jiwer.wer(
                    resampled_text,
                    processed_text,
                    truth_transform=transforms,
                    hypothesis_transform=transforms,
                )
                wer_scores.append(wer)
                file_results.append({
                    'filename': filename,
                    'resampled_text': resampled_text,
                    'processed_text': processed_text,
                    'wer': wer
                })
                
                print(f"WER: {wer:.4f}")
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        # Calculate and display mean WER for the speaker
        if wer_scores:
            mean_wer = np.mean(wer_scores)
            speaker_result = {
                'mean_wer': mean_wer,
                'min_wer': min(wer_scores),
                'max_wer': max(wer_scores),
                'file_count': len(wer_scores),
                'files': file_results
            }
            all_speaker_results[speaker] = speaker_result
            
            print(f"\n===== Speaker: {speaker} Summary =====")
            print(f"Mean WER: {mean_wer:.4f}")
            print(f"Min WER: {min(wer_scores):.4f}")
            print(f"Max WER: {max(wer_scores):.4f}")
            print(f"Number of files: {len(wer_scores)}")
        else:
            print(f"No WER scores calculated for {speaker}")
    
    # Print overall summary
    if all_speaker_results:
        print("\n===== Overall Summary =====")
        all_wers = [result['mean_wer'] for result in all_speaker_results.values()]
        print(f"Overall Mean WER across all speakers: {np.mean(all_wers):.4f}")
        
        # Print speaker ranking by WER
        print("\nSpeaker Ranking (by Mean WER):")
        for i, (speaker, data) in enumerate(sorted(all_speaker_results.items(), key=lambda x: x[1]['mean_wer'])):
            print(f"{i+1}. {speaker}: {data['mean_wer']:.4f} (Files: {data['file_count']})")
    
    return all_speaker_results

if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    
    generator = VoiceModel()
    
    kwargs = {
        "folder": args.model,
        "map_location": args.device,
        "package": False
    }
    
    generator, g_extra = generator.load_from_folder(**kwargs)
    generator.to(args.device)
    generator.eval()
    
    print(f"Model loaded from: {args.model}")

    targets = ['p227', 'p228']
    speaker_embeddings, speaker_means, speaker_stds = get_speaker_embeddings(targets)
    
    shutil.rmtree(args.resampled_audio_folder)
    shutil.rmtree(args.processed_audio_folder)

    process_audio_files(generator,
                        targets,
                        speaker_embeddings,
                        speaker_means,
                        speaker_stds,
                        args.input_audio_folder,
                        args.resampled_audio_folder,
                        args.processed_audio_folder,
                        args.target_sr)

    # Calculate similarity scores
    print("\nCalculating similarity scores...")
    similarity_results = calculate_similarity_scores(
        args.processed_audio_folder,
        args.target_speaker_folder,
        targets,
    )

    print_similarity_report(similarity_results)

    # Calculate WER
    calculate_wer(targets, args.resampled_audio_folder, args.processed_audio_folder)