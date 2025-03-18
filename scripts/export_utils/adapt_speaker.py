import os
import librosa
import torch
from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda")

def extract_utterance_fcpe(y, sr: int, frame_len_samples: int):
    f0_target_length=(y.shape[-1] // frame_len_samples)
    f0 = pitch_model.infer(y.unsqueeze(-1),
                           sr=sr,
                           decoder_mode='local_argmax',
                           threshold=0.006,
                           f0_min=50,
                           f0_max=550,
                           interp_uv=False,
                           output_interp_target_length=f0_target_length)
    return f0

def get_f0_fcpe(x, fs: int, win_length: int):
    f0 = extract_utterance_fcpe(x, fs, win_length)
    return f0

def extract_f0_mean_std(f0s: torch.Tensor):
    f0s = f0s[~torch.isnan(f0s)]
    f0s = f0s[f0s > 0]
    f0s_mean = torch.mean(f0s)
    return f0s_mean

def adapt_speaker(folder, speaker_encoder, pqmf):
    f0_means = []
    embeddings = []
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.wav', '.flac')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                audio, sr = librosa.load(file_path, sr=44100, mono=True)
                audio = torch.tensor(audio[:131072]).unsqueeze(0)
                
                if audio.shape[-1] < 131072:
                    pad = torch.zeros(1, 131072-audio.shape[-1])
                    audio = torch.cat((audio, pad), dim=-1)
                    
                f0 = get_f0_fcpe(audio, sr, 2048)
                f0_mean = extract_f0_mean_std(f0)
                audio_multiband = pqmf(audio.unsqueeze(0))
                emb = speaker_encoder(audio_multiband)

                if not torch.isnan(f0_mean):
                    f0_means.append(f0_mean) 
                    embeddings.append(emb)
    
    # Calculate mean of f0 values and embeddings
    f0_val = torch.stack(f0_means).mean()
    embedding_val = torch.stack(embeddings).mean(dim=0)
    
    return f0_val.detach().cpu().item(), embedding_val.cpu()