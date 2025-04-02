import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import json
import os
import torch.nn as nn
import argparse
from typing import Optional
import torch.nn.functional as F
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
    f0s_std = torch.std(f0s)
    return f0s_mean, f0s_std

def entropy(logits: torch.Tensor):
    """Entropy-based periodicity - Low entropy indicates high periodicity"""
    distribution = torch.nn.functional.softmax(logits, dim=1)
    return (
        1 + 1 / math.log(1440) * \
        (distribution * torch.log(distribution + 1e-7)).sum(dim=1))

def threshold(periodicity: torch.Tensor, value: float=0.065):
    """Threshold periodicity to produce voiced/unvoiced classifications"""
    return periodicity > value

def bins_to_cents(bins: torch.Tensor):
    """Converts pitch bins to cents"""
    return 5.0 * bins

def bins_to_frequency(bins: torch.Tensor):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))

def cents_to_frequency(cents: torch.Tensor):
    """Converts cents to frequency in Hz"""
    return 31.0 * 2 ** (cents / 1200)

def cents(a: torch.Tensor, b: torch.Tensor):
    """Compute pitch difference in cents"""
    return 1200 * torch.log2(a / b)

def a_weighting(frequencies, min_db: torch.Tensor = torch.tensor([-80])):
    f_sq = frequencies ** 2.0

    const = torch.tensor([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights = 2.0 + 20.0 * (
        torch.log10(const[0])
        + 2 * torch.log10(f_sq)
        - torch.log10(f_sq + const[0])
        - torch.log10(f_sq + const[1])
        - 0.5 * torch.log10(f_sq + const[2])
        - 0.5 * torch.log10(f_sq + const[3])
    )

    if min_db is None:
        return weights
    else:
        return torch.maximum(min_db, weights)

def extract_loudness(signal, sr: int, block_size: int=1024, n_fft: int=1024):
    S = torch.stft(
        signal.squeeze(1),
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
        return_complex=True)
    
    S = torch.log(abs(S) + 1e-7)
    f = torch.linspace(0, sr // 2, (1 + n_fft // 2))
    a_weight = a_weighting(f).to(signal.device)

    S = S + a_weight.reshape(-1, 1)
    S = torch.mean(S, 1)[..., :-1]

    return S

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)

def extract_rms(signal: torch.Tensor, frame_size: int, hop_size: Optional[int] = None, upsample: Optional[str] = True) -> torch.Tensor:

    if hop_size is None:
        hop_size = frame_size
        
    batch_size, _, signal_length = signal.shape
    signal = signal.reshape(batch_size, signal_length)
    frames = signal.unfold(1, frame_size, hop_size)
    frames_squared = torch.square(frames)
    mean_squared = torch.mean(frames_squared, dim=2)
    rms_values = torch.sqrt(mean_squared)

    if upsample:
        rms_values = upsample(rms_values.unsqueeze(-1), frame_size)
        return rms_values.transpose(2,1)
    else:
        return rms_values
