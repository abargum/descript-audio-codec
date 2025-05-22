"""
Word Error Rate (WER) and Character Error Rate (CER) evaluation using Whisper ASR.
"""

import os
import numpy as np
import torch
import jiwer
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def setup_whisper_pipeline():
    """
    Set up the Whisper ASR pipeline.
    
    Returns:
        pipeline: Initialized Whisper ASR pipeline
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    try:
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
            device=device
        )
        return pipe
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

def setup_text_transforms():
    """
    Set up jiwer text transforms for normalization.
    
    Returns:
        jiwer.Compose: Text transformation pipeline
    """
    transforms = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    return transforms

def calculate_wer_cer_for_speaker(speaker, resampled_speaker_path, processed_speaker_path, pipe, transforms):
    """
    Calculate WER and CER for a single speaker.
    
    Args:
        speaker (str): Speaker ID
        resampled_speaker_path (str): Path to resampled audio files
        processed_speaker_path (str): Path to processed audio files
        pipe: Whisper ASR pipeline
        transforms: Text normalization transforms
    
    Returns:
        dict: Results containing WER, CER, and file details
    """
    print(f"\n===== Processing speaker: {speaker} =====")
    
    # Check if both paths exist
    if not os.path.exists(resampled_speaker_path):
        print(f"Resampled folder for {speaker} not found at {resampled_speaker_path}")
        return None
    if not os.path.exists(processed_speaker_path):
        print(f"Processed folder for {speaker} not found at {processed_speaker_path}")
        return None
    
    # Create dictionaries mapping filenames to full paths
    resampled_files = {f.name: f for f in Path(resampled_speaker_path).glob("**/*.wav")}
    processed_files = {f.name: f for f in Path(processed_speaker_path).glob("**/*.wav")}
    
    # Find files that exist in both folders (same filename)
    common_filenames = set(resampled_files.keys()) & set(processed_files.keys())
    
    if not common_filenames:
        print(f"No matching filenames found for speaker {speaker}")
        return None
    
    print(f"Found {len(common_filenames)} matching files for speaker {speaker}")
    
    # Calculate WER and CER for each matching file
    wer_scores = []
    cer_scores = []
    file_results = []
    
    for filename in tqdm(sorted(common_filenames), desc=f"Calculating WER/CER for {speaker}", leave=False):
        resampled_file = str(resampled_files[filename])
        processed_file = str(processed_files[filename])
        
        try:
            # Get ASR results for both versions
            resampled_result = pipe(resampled_file)
            processed_result = pipe(processed_file)
            
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
            
            # Calculate CER (clean punctuation for better comparison)
            resampled_text_clean = resampled_text.replace(",", "").replace(".", "").lower()
            processed_text_clean = processed_text.replace(",", "").replace(".", "").lower()
            cer = jiwer.cer(resampled_text_clean, processed_text_clean)
            cer_scores.append(cer)
            
            file_results.append({
                'filename': filename,
                'resampled_text': resampled_text,
                'processed_text': processed_text,
                'wer': wer,
                'cer': cer
            })
            
            print(f"WER: {wer:.4f}")
            print(f"CER: {cer:.4f}")
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
    
    # Calculate and display mean scores for the speaker
    if wer_scores and cer_scores:
        mean_wer = np.mean(wer_scores)
        mean_cer = np.mean(cer_scores)
        
        speaker_result = {
            'mean_wer': mean_wer,
            'mean_cer': mean_cer,
            'min_wer': min(wer_scores),
            'max_wer': max(wer_scores),
            'min_cer': min(cer_scores),
            'max_cer': max(cer_scores),
            'file_count': len(wer_scores),
            'files': file_results
        }
        
        print(f"\n===== Speaker: {speaker} Summary =====")
        print(f"Mean WER: {mean_wer:.4f}")
        print(f"Mean CER: {mean_cer:.4f}")
        print(f"Min WER: {min(wer_scores):.4f}")
        print(f"Max WER: {max(wer_scores):.4f}")
        print(f"Min CER: {min(cer_scores):.4f}")
        print(f"Max CER: {max(cer_scores):.4f}")
        print(f"Number of files: {len(wer_scores)}")
        
        return speaker_result
    else:
        print(f"No WER/CER scores calculated for {speaker}")
        return None

def calculate_wer_cer(targets, resampled_audio_folder, processed_audio_folder):
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER) for all target speakers.
    
    Args:
        targets (list): List of target speaker IDs
        resampled_audio_folder (str): Path to resampled audio folder
        processed_audio_folder (str): Path to processed audio folder
    
    Returns:
        dict: Dictionary containing WER and CER results for each speaker
    """
    # Set up Whisper pipeline
    pipe = setup_whisper_pipeline()
    if pipe is None:
        print("Failed to initialize Whisper pipeline")
        return {}
    
    # Set up text transforms
    transforms = setup_text_transforms()
    
    # Store all speaker results for final summary
    all_speaker_results = {}
    
    for speaker in tqdm(targets, desc="Processing speakers"):
        # Get speaker paths
        resampled_speaker_path = os.path.join(resampled_audio_folder, speaker)
        processed_speaker_path = os.path.join(processed_audio_folder, speaker)
        
        # Calculate WER and CER for this speaker
        speaker_result = calculate_wer_cer_for_speaker(
            speaker=speaker,
            resampled_speaker_path=resampled_speaker_path,
            processed_speaker_path=processed_speaker_path,
            pipe=pipe,
            transforms=transforms
        )
        
        if speaker_result:
            all_speaker_results[speaker] = speaker_result
    
    # Print overall summary
    if all_speaker_results:
        print("\n===== Overall WER/CER Summary =====")
        all_wers = [result['mean_wer'] for result in all_speaker_results.values()]
        all_cers = [result['mean_cer'] for result in all_speaker_results.values()]
        
        print(f"Overall Mean WER across all speakers: {np.mean(all_wers):.4f}")
        print(f"Overall Mean CER across all speakers: {np.mean(all_cers):.4f}")
        
        # Print speaker ranking by WER
        print("\nSpeaker Ranking (by Mean WER):")
        for i, (speaker, data) in enumerate(sorted(all_speaker_results.items(), key=lambda x: x[1]['mean_wer'])):
            print(f"{i+1}. {speaker}: WER={data['mean_wer']:.4f}, CER={data['mean_cer']:.4f} (Files: {data['file_count']})")
    
    return all_speaker_results

def print_wer_cer_report(wer_cer_results):
    """Print a formatted WER/CER report."""
    print("\n===== WER/CER EVALUATION REPORT =====\n")
    
    for speaker, data in wer_cer_results.items():
        print(f"\n== Speaker: {speaker} ==")
        print(f"Mean WER: {data['mean_wer']:.4f}")
        print(f"Mean CER: {data['mean_cer']:.4f}")
        print(f"Files evaluated: {data['file_count']}")
    
    print("\n===========================================")