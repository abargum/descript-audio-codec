"""
Generate comprehensive evaluation reports with all metrics.
"""
import os
import datetime
import random
import string

def calculate_overall_metrics(similarity_results, dnsmos_results, wer_cer_results):
    """
    Calculate overall metrics across all speakers.
    
    Args:
        similarity_results (dict): Voice similarity results
        dnsmos_results (dict): DNSMOS quality results
        wer_cer_results (dict): WER/CER results
    
    Returns:
        dict: Overall metrics summary
    """
    overall_metrics = {}
    
    # Similarity metrics
    if similarity_results:
        avg_similarities = [data['average_similarity'] for speaker, data in similarity_results.items() 
                           if speaker != 'cross_speaker_similarity']
        if avg_similarities:
            overall_metrics['overall_similarity'] = sum(avg_similarities) / len(avg_similarities)
    
    # DNSMOS metrics
    if dnsmos_results:
        all_bak = [data['mean_bak'] for data in dnsmos_results.values()]
        all_sig = [data['mean_sig'] for data in dnsmos_results.values()]
        all_ovrl = [data['mean_ovrl'] for data in dnsmos_results.values()]
        
        if all_bak and all_sig and all_ovrl:
            overall_metrics['overall_dnsmos_bak'] = sum(all_bak) / len(all_bak)
            overall_metrics['overall_dnsmos_sig'] = sum(all_sig) / len(all_sig)
            overall_metrics['overall_dnsmos_ovrl'] = sum(all_ovrl) / len(all_ovrl)
    
    # WER/CER metrics
    if wer_cer_results:
        all_wers = [data['mean_wer'] for data in wer_cer_results.values()]
        all_cers = [data['mean_cer'] for data in wer_cer_results.values()]

        all_corpus_wers = [data['corpus_wer'] for data in wer_cer_results.values()]
        all_corpus_cers = [data['corpus_cer'] for data in wer_cer_results.values()]
        
        if all_wers:
            overall_metrics['overall_wer'] = sum(all_wers) / len(all_wers)
        if all_cers:
            overall_metrics['overall_cer'] = sum(all_cers) / len(all_cers)
        if all_corpus_wers:
            overall_metrics['corpus_wer'] = sum(all_corpus_wers) / len(all_corpus_wers)
        if all_corpus_cers:
            overall_metrics['corpus_cer'] = sum(all_corpus_cers) / len(all_corpus_cers)
    
    return overall_metrics

def write_similarity_metrics(f, similarity_results):
    """Write similarity metrics to file."""
    f.write("=== VOICE SIMILARITY METRICS ===\n")
    for speaker, data in similarity_results.items():
        if speaker != 'cross_speaker_similarity':
            f.write(f"\nSpeaker: {speaker}\n")
            f.write(f"Average similarity score: {data['average_similarity']:.4f}\n")
            
            # Add top 3 and bottom 3 files if there are enough samples
            if len(data['individual_scores']) >= 6:
                f.write("\nTop 3 most similar files:\n")
                for i, (filename, score) in enumerate(data['individual_scores'][:3]):
                    f.write(f"{i+1}. {filename}: {score:.4f}\n")
                
                f.write("\nBottom 3 least similar files:\n")
                for i, (filename, score) in enumerate(data['individual_scores'][-3:]):
                    f.write(f"{i+1}. {filename}: {score:.4f}\n")

def write_dnsmos_metrics(f, dnsmos_results):
    """Write DNSMOS metrics to file."""
    f.write("\n\n=== DNSMOS METRICS ===\n")
    
    for speaker, data in dnsmos_results.items():
        f.write(f"\nSpeaker: {speaker}\n")
        f.write(f"BAK (background quality): {data['mean_bak']:.4f}\n")
        f.write(f"SIG (signal quality): {data['mean_sig']:.4f}\n")
        f.write(f"OVRL (overall quality): {data['mean_ovrl']:.4f}\n")

def write_wer_cer_metrics(f, wer_cer_results):
    """Write WER/CER metrics to file."""
    f.write("\n\n=== WORD ERROR RATE (WER) AND CHARACTER ERROR RATE (CER) METRICS ===\n")
    
    for speaker, data in wer_cer_results.items():
        f.write(f"\nSpeaker: {speaker}\n")
        f.write(f"Mean WER: {data['mean_wer']:.4f}\n")
        f.write(f"Mean CER: {data['mean_cer']:.4f}\n")
        f.write(f"Corpus WER: {data['corpus_wer']:.4f}\n")
        f.write(f"Corpus CER: {data['corpus_cer']:.4f}\n")
        f.write(f"Min WER: {data['min_wer']:.4f}\n")
        f.write(f"Max WER: {data['max_wer']:.4f}\n")
        f.write(f"Min CER: {data['min_cer']:.4f}\n")
        f.write(f"Max CER: {data['max_cer']:.4f}\n")
        f.write(f"Files evaluated: {data['file_count']}\n")

def write_overall_summary(f, overall_metrics):
    """Write overall summary metrics to file."""
    f.write("\n\n=== OVERALL METRICS SUMMARY ===\n")
    
    if 'overall_similarity' in overall_metrics:
        f.write(f"Overall Average Voice Similarity: {overall_metrics['overall_similarity']:.4f}\n")
    
    if 'overall_dnsmos_bak' in overall_metrics:
        f.write(f"Overall Average DNSMOS BAK: {overall_metrics['overall_dnsmos_bak']:.4f}\n")
        f.write(f"Overall Average DNSMOS SIG: {overall_metrics['overall_dnsmos_sig']:.4f}\n")
        f.write(f"Overall Average DNSMOS OVRL: {overall_metrics['overall_dnsmos_ovrl']:.4f}\n")
    
    if 'overall_wer' in overall_metrics:
        f.write(f"Overall Average WER: {overall_metrics['overall_wer']:.4f}\n")
    
    if 'overall_cer' in overall_metrics:
        f.write(f"Overall Average CER: {overall_metrics['overall_cer']:.4f}\n")

def save_metrics_to_file(args, similarity_results, dnsmos_results, wer_cer_results=None):
    """
    Save all metrics to a comprehensive text file.
    
    Args:
        args: The command line arguments
        similarity_results: Results from calculate_similarity_scores
        dnsmos_results: Results from calculate_dnsmos_scores
        wer_cer_results: Results from calculate_wer_cer (optional)
    
    Returns:
        str: Path to the saved metrics file
    """
    # Extract model name from the model path
    model_path = args.model.rstrip('/')  # Remove trailing slash if present
    model_name = os.path.basename(model_path)
    if not model_name:  # If basename is empty, try the parent directory
        model_name = os.path.basename(os.path.dirname(model_path))

    data_path = args.input_audio_folder.rstrip('/')  # Remove trailing slash if present
    
    # Create metrics folder next to processed audio folder
    base_dir = os.path.dirname(args.processed_audio_folder)
    metrics_folder = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Create metrics file path
    chars = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    random_chars = ''.join(random.choices(chars, k=4))
    metrics_file = os.path.join(metrics_folder, f"{model_name}_evaluation_report_{random_chars}.txt")
    
    # Current datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(similarity_results, dnsmos_results, wer_cer_results)
    
    # Write metrics to file
    with open(metrics_file, 'w') as f:
        # Header
        f.write(f"=== VOICE MODEL EVALUATION REPORT: {model_name} ===\n")
        f.write(f"Evaluation date: {now}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Target speakers: {', '.join(args.targets)}\n")
        f.write(f"Device used: {args.device}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        
        # Write individual metric sections
        if similarity_results:
            write_similarity_metrics(f, similarity_results)
        
        if dnsmos_results:
            write_dnsmos_metrics(f, dnsmos_results)
        
        if wer_cer_results:
            write_wer_cer_metrics(f, wer_cer_results)
        
        # Write overall summary
        write_overall_summary(f, overall_metrics)
        
        f.write("\n=== END OF EVALUATION REPORT ===\n")
    
    print(f"\nComprehensive evaluation report saved to: {metrics_file}")
    
    # Print summary to console
    print("\n=== EVALUATION SUMMARY ===")
    for key, value in overall_metrics.items():
        metric_name = key.replace('overall_', '').replace('_', ' ').title()
        print(f"{metric_name}: {value:.4f}")
    
    return metrics_file

def print_all_reports(similarity_results, dnsmos_results, wer_cer_results):
    """Print all evaluation reports to console."""
    from evaluation.similarity_evaluator import print_similarity_report
    from evaluation.dnsmos_evaluator import print_dnsmos_report
    from evaluation.wer_evaluator import print_wer_cer_report
    
    if similarity_results:
        print_similarity_report(similarity_results)
    
    if dnsmos_results:
        print_dnsmos_report(dnsmos_results)
    
    if wer_cer_results:
        print_wer_cer_report(wer_cer_results)