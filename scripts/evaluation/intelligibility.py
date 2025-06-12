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
    print(f"\n===== Processing speaker: {speaker} =====")

    if not os.path.exists(resampled_speaker_path):
        print(f"Resampled folder for {speaker} not found at {resampled_speaker_path}")
        return None
    if not os.path.exists(processed_speaker_path):
        print(f"Processed folder for {speaker} not found at {processed_speaker_path}")
        return None

    resampled_files = {f.name: f for f in Path(resampled_speaker_path).glob("**/*.wav")}
    processed_files = {f.name: f for f in Path(processed_speaker_path).glob("**/*.wav")}

    common_filenames = set(resampled_files.keys()) & set(processed_files.keys())

    if not common_filenames:
        print(f"No matching filenames found for speaker {speaker}")
        return None

    print(f"Found {len(common_filenames)} matching files for speaker {speaker}")

    wer_scores = []
    cer_scores = []
    file_results = []
    all_resampled_texts = []
    all_processed_texts = []

    for filename in tqdm(sorted(common_filenames), desc=f"Calculating WER/CER for {speaker}", leave=False):
        resampled_file = str(resampled_files[filename])
        processed_file = str(processed_files[filename])

        try:
            resampled_result = pipe(resampled_file)
            processed_result = pipe(processed_file)

            resampled_text = resampled_result['text']
            processed_text = processed_result['text']

            print(f"\nFile: {filename}")
            print(f"Resampled: {resampled_text}")
            print(f"Processed: {processed_text}")

            wer = jiwer.wer(
                resampled_text,
                processed_text,
                truth_transform=transforms,
                hypothesis_transform=transforms,
            )
            wer_scores.append(wer)

            resampled_text_clean = resampled_text.replace(",", "").replace(".", "").lower()
            processed_text_clean = processed_text.replace(",", "").replace(".", "").lower()
            cer = jiwer.cer(resampled_text_clean, processed_text_clean)
            cer_scores.append(cer)

            all_resampled_texts.append(resampled_text)
            all_processed_texts.append(processed_text)

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

    if wer_scores and cer_scores:
        mean_wer = np.mean(wer_scores)
        mean_cer = np.mean(cer_scores)

        resampled_corpus = " ".join(all_resampled_texts)
        processed_corpus = " ".join(all_processed_texts)

        corpus_wer = jiwer.wer(
            resampled_corpus,
            processed_corpus,
            truth_transform=transforms,
            hypothesis_transform=transforms,
        )

        resampled_clean = resampled_corpus.replace(",", "").replace(".", "").lower()
        processed_clean = processed_corpus.replace(",", "").replace(".", "").lower()
        corpus_cer = jiwer.cer(resampled_clean, processed_clean)

        speaker_result = {
            'mean_wer': mean_wer,
            'mean_cer': mean_cer,
            'corpus_wer': corpus_wer,
            'corpus_cer': corpus_cer,
            'min_wer': min(wer_scores),
            'max_wer': max(wer_scores),
            'min_cer': min(cer_scores),
            'max_cer': max(cer_scores),
            'file_count': len(wer_scores),
            'files': file_results
        }

        print(f"\n===== Speaker: {speaker} Summary =====")
        print(f"Mean WER: {mean_wer:.4f}")
        print(f"Corpus-level WER: {corpus_wer:.4f}")
        print(f"Mean CER: {mean_cer:.4f}")
        print(f"Corpus-level CER: {corpus_cer:.4f}")
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
    pipe = setup_whisper_pipeline()
    if pipe is None:
        print("Failed to initialize Whisper pipeline")
        return {}

    transforms = setup_text_transforms()
    all_speaker_results = {}

    for speaker in tqdm(targets, desc="Processing speakers"):
        resampled_speaker_path = os.path.join(resampled_audio_folder, speaker)
        processed_speaker_path = os.path.join(processed_audio_folder, speaker)

        speaker_result = calculate_wer_cer_for_speaker(
            speaker=speaker,
            resampled_speaker_path=resampled_speaker_path,
            processed_speaker_path=processed_speaker_path,
            pipe=pipe,
            transforms=transforms
        )

        if speaker_result:
            all_speaker_results[speaker] = speaker_result

    if all_speaker_results:
        print("\n===== Overall WER/CER Summary =====")
        all_wers = [result['mean_wer'] for result in all_speaker_results.values()]
        all_cers = [result['mean_cer'] for result in all_speaker_results.values()]

        all_wers_corpus = [result['corpus_wer'] for result in all_speaker_results.values()]
        all_cers_corpus = [result['corpus_cer'] for result in all_speaker_results.values()]

        print(f"Overall Mean WER across all speakers: {np.mean(all_wers):.4f}")
        print(f"Overall Mean CER across all speakers: {np.mean(all_cers):.4f}")
        print(f"Overall Corpus WER across all speakers: {np.mean(all_wers_corpus):.4f}")
        print(f"Overall Corpus CER across all speakers: {np.mean(all_cers_corpus):.4f}")

        print("\nSpeaker Ranking (by Mean WER):")
        for i, (speaker, data) in enumerate(sorted(all_speaker_results.items(), key=lambda x: x[1]['mean_wer'])):
            print(f"{i+1}. {speaker}: WER={data['mean_wer']:.4f}, CER={data['mean_cer']:.4f} (Files: {data['file_count']})")

        print("\nSpeaker Ranking (by Corpus WER):")
        for i, (speaker, data) in enumerate(sorted(all_speaker_results.items(), key=lambda x: x[1]['corpus_wer'])):
            print(f"{i+1}. {speaker}: C-WER={data['corpus_wer']:.4f}, C-CER={data['corpus_cer']:.4f} (Files: {data['file_count']})")

    return all_speaker_results

def print_wer_cer_report(wer_cer_results):
    print("\n===== WER/CER EVALUATION REPORT =====\n")

    for speaker, data in wer_cer_results.items():
        print(f"\n== Speaker: {speaker} ==")
        print(f"Mean WER: {data['mean_wer']:.4f}")
        print(f"Corpus-level WER: {data['corpus_wer']:.4f}")
        print(f"Mean CER: {data['mean_cer']:.4f}")
        print(f"Corpus-level CER: {data['corpus_cer']:.4f}")
        print(f"Files evaluated: {data['file_count']}")

    print("\n===========================================")
