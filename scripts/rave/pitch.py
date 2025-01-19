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