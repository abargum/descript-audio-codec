import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import shutil
from pathlib import Path

from utils import set_seed, load_voice_model, get_speaker_embeddings
from audio_process import process_audio_files
from similarity import calculate_similarity_scores
from dnsmos import calculate_dnsmos_scores
from intelligibility import calculate_wer_cer
from report import save_metrics_to_file

def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description='Voice model processing script')
    parser.add_argument('--model', type=str, required=True, help='Path to the model folder')
    parser.add_argument('--input_audio_folder', type=str, default="audio", 
                       help='Path to the input audio folder')
    parser.add_argument('--resampled_audio_folder', type=str, default="scripts/evaluation/resampled", 
                       help='Path to the resampled input audio folder')
    parser.add_argument('--processed_audio_folder', type=str, default="scripts/evaluation/processed",  
                       help='Path to the processed audio folder')
    parser.add_argument('--target_speaker_folder', type=str, default="vctk-small",  
                       help='Path to the target audio folder')
    parser.add_argument('--device', type=str, default="cuda", 
                       help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--targets', nargs='+', default=['p228'],
                       help='List of target speakers')
    return parser

def cleanup_previous_results(args):
    """Clean up previous evaluation results."""
    processed_folder_44 = args.processed_audio_folder + "_44"
    
    folders_to_clean = [
        args.resampled_audio_folder,
        args.processed_audio_folder,
        processed_folder_44
    ]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"Cleaning up: {folder}")
            shutil.rmtree(folder)

def main():
    """Main evaluation pipeline."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set random seed
    #set_seed(args.seed)
    
    # Load model
    print("Loading voice model...")
    generator = load_voice_model(args.model, args.device)
    print(f"Model loaded from: {args.model}")
    
    # Get speaker embeddings
    print("Loading speaker embeddings...")
    speaker_embeddings, speaker_means, speaker_stds = get_speaker_embeddings(args.targets)
    
    # Clean up previous results
    cleanup_previous_results(args)
    
    # Process audio files
    print("\nProcessing audio files...")
    process_audio_files(
        generator=generator,
        targets=args.targets,
        embeddings=speaker_embeddings,
        means=speaker_means,
        stds=speaker_stds,
        input_folder=args.input_audio_folder,
        output_folder=args.resampled_audio_folder,
        processed_folder=args.processed_audio_folder,
        device=args.device
    )
    
    # Calculate similarity scores
    print("\nCalculating similarity scores...")
    similarity_results = calculate_similarity_scores(
        processed_folder=args.processed_audio_folder,
        target_speaker_folder=args.target_speaker_folder,
        targets=args.targets
    )
    
    # Calculate DNSMOS scores
    print("\nCalculating DNSMOS scores...")
    processed_folder_44 = args.processed_audio_folder + "_44"
    dnsmos_results = calculate_dnsmos_scores(
        processed_folder_44=processed_folder_44,
        targets=args.targets
    )
    
    # Calculate WER and CER
    print("\nCalculating WER and CER...")
    wer_cer_results = calculate_wer_cer(
        targets=args.targets,
        resampled_audio_folder=args.resampled_audio_folder,
        processed_audio_folder=args.processed_audio_folder
    )
    
    # Save metrics to file
    print("\nSaving metrics to file...")
    metrics_file = save_metrics_to_file(
        args=args,
        similarity_results=similarity_results,
        dnsmos_results=dnsmos_results,
        wer_cer_results=wer_cer_results
    )
    
    print(f"\nEvaluation complete! Results saved to: {metrics_file}")

if __name__ == "__main__":
    main()