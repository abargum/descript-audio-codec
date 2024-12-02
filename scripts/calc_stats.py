import os
import torch
import librosa

from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda:0")

def calculate_speaker_f0_stats(folder_path, sample_rate=16000):
    all_f0_values = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.wav', '.mp3', 'flac')):
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=sample_rate)
            
            # Convert to PyTorch tensor
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            
            # Extract F0 (use your pitch extractor here)
            f0 = pitch_model.infer(audio_tensor, sr=sr, decoder_mode='local_argmax', threshold=0.006, f0_min=50, f0_max=550)
            # Remove unvoiced frames (F0 = 0)
            f0_nonzero = f0[f0 > 0]
            all_f0_values.append(f0_nonzero)

    # Concatenate all F0 values and calculate global stats
    all_f0_values = torch.cat(all_f0_values)
    global_mean = torch.mean(all_f0_values).item()
    global_std = torch.std(all_f0_values).item()

    return global_mean, global_std

def calculate_all_speakers_stats(root_folder, sample_rate=16000):
    speaker_stats = {}

    for speaker_folder in os.listdir(root_folder):
        speaker_path = os.path.join(root_folder, speaker_folder)
        if os.path.isdir(speaker_path):  # Ensure it's a folder
            print(f"Processing speaker: {speaker_folder}")
            mean_f0, std_f0 = calculate_speaker_f0_stats(speaker_path, sample_rate)
            speaker_stats[speaker_folder] = (mean_f0, std_f0)

    return speaker_stats

# Example usage
root_folder = "../vctk-small"
all_speaker_stats = calculate_all_speakers_stats(root_folder)

# Print results
for speaker, stats in all_speaker_stats.items():
    print(f"Speaker: {speaker}, Mean F0: {stats[0]:.2f} Hz, Std F0: {stats[1]:.2f} Hz")
