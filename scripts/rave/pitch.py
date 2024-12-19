import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import json
import os
import argparse
import torch.nn.functional as F
from torchfcpe import spawn_bundled_infer_model

pitch_model = spawn_bundled_infer_model(device="cuda:0")

def extract_utterance_fcpe(y, sr: int, frame_len_samples: int) -> torch.Tensor:
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

def get_f0_fcpe(x, fs: int, win_length: int) -> torch.Tensor:
    f0 = extract_utterance_fcpe(x, fs, win_length)
    return f0

def extract_f0_mean_std(f0s: torch.Tensor):
    f0s = f0s[~torch.isnan(f0s)]
    f0s = f0s[f0s > 0]
    f0s_mean = torch.mean(f0s)
    f0s_std = torch.std(f0s)
    return f0s_mean, f0s_std

def get_log_norm_f0(f0s: torch.Tensor) -> torch.Tensor:
    index_nonzero = (f0s > 0)
    log_f0 = torch.log(f0s)
    mean = torch.mean(log_f0[index_nonzero])
    std = torch.std(log_f0[index_nonzero])

    f0s[index_nonzero] = (log_f0[index_nonzero] - mean) / std / 4.0
    f0s[index_nonzero] = (f0s[index_nonzero] + 1.0) / 2.0
    return f0s.unsqueeze(1)

def get_uv(f0: torch.Tensor, voiced_threshold: float = 0.0) -> torch.Tensor:
    uv = torch.ones_like(f0)
    uv = uv * (f0 > voiced_threshold)
    return uv.unsqueeze(1)

def frame(audio: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
        audio_np = audio.numpy()
        audio_np = np.pad(audio_np, (0, frame_length // 2), mode="constant")
        frames = librosa.util.frame(audio_np, frame_length=frame_length, hop_length=hop_length)
        return torch.tensor(frames)

def get_energy(audio: torch.Tensor, block_size: int = 1024):
    frames = []
    for i in range(audio.shape[-1] // block_size):
        frame = audio[:, (i*block_size):((i+1)*block_size)]
        energy_val = torch.pow(frame, 2).sum(axis=-1, keepdim=True)
        frames.append(energy_val)
    frames = torch.cat(frames, dim=-1)
    frames = frames / torch.max(frames)
    return frames.unsqueeze(1)

